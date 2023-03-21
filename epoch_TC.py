import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from random import seed


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, classes, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.num_classes = len(classes)

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y, a, b, c, d, e, epoch, device):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, epoch):

        self.on_epoch_start()

        if self.stage_name == 'train':
            seed(1)

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        IoU = {'intersection': [0] * self.num_classes, 'union': [0] * self.num_classes, 'inconsistencies': []} # parametrizzarlo!!!

        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        states = None
        t = 1
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            num_proc_old = -1
            self.model.segmentationModel.eval()
            init_states = True
            for x, y, phase, elap, i, num_proc in iterator:
                if self.stage_name == 'train':
                    init_states = True
                if self.stage_name != 'train':
                    # so in the test we suppose that all the sequences are in order between each other
                    if num_proc != num_proc_old:
                        init_states = True
                        num_proc_old = num_proc

                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred, IoU = self.batch_update(x, y, phase, elap, i, IoU, init_states, epoch, self.device)

                init_states = False
                t += 1

                if loss is not None:
                    # update loss logs
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {'loss': loss_meter.mean}
                    logs.update(loss_logs)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)


            if IoU is not None:
                IoU['union'] = [elem + 1e-8 for elem in IoU['union']]
                valid_IoU = np.divide(IoU['intersection'], IoU['union'])
                print(np.mean(IoU['inconsistencies']))

            else:
                return logs

        return logs, valid_IoU, np.mean(IoU['inconsistencies'])

class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, classes, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            classes=classes,
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.old_prediction = 0
        self.old_prediction_correct = 0
        self.old_gt = 0

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, phase_target, elap, i, IoU, init_states, epoch, device):

        prediction, loss, tc_loss = self.model.forward(x, y, elap, x.shape[0], reset_states=init_states,
                                                       optimizer=self.optimizer)
        preds = torch.argmax(prediction, dim=1)

        #img = class2color(preds[0].cpu().numpy())
        #cv2.imwrite('/workspace/ResultsConvLSTM_' + 'train' + '.png', img[:, :, ::-1])

        self.old_prediction = prediction.data
        self.old_gt = (torch.argmax(y[:, -1], dim=1)).data
        self.old_prediction_correct = torch.argmax(prediction, dim=1) == self.old_gt

        return loss, prediction, None


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, classes, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            classes=classes,
            device=device,
            verbose=verbose,
        )
        self.inconsistencies = 0
        self.old_prediction = None
        self.num_classes = len(classes)

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, phase_target, elap, ix, IoU, init_states, epoch, device):
        # image time t, mask time t
        with torch.no_grad():
            prediction, loss, tc_loss = self.model.forward(x, torch.argmax(y, dim=1), elap, x.shape[0],
                                                       reset_states=init_states, optimizer=None, compute_loss=False)

            loss = self.loss(prediction, torch.argmax(y[:, -1], dim=1).long())
            preds = torch.argmax(prediction, dim=1)

           # img = class2color(preds[0].cpu().numpy())
            #cv2.imwrite('/workspace/valid/' + str(int(ix)) + '.png', img[:, :, ::-1])
            gts = torch.argmax(y[:, -1], dim=1)

            #print('y shape: ', y.shape)
            #print('y[:, -1] shape: ', y[:, -1].shape)

            i = []
            u = []
            for e in range(self.num_classes):
                A = preds == e
                B = gts == e
                i.append(torch.bitwise_and(A, B).int().sum().cpu().numpy().min())
                u.append(torch.bitwise_or(A, B).int().sum().cpu().numpy().min())

            IoU['intersection'] = [sum(x) for x in zip(IoU['intersection'], i)]
            IoU['union'] = [sum(x) for x in zip(IoU['union'], u)]
            if ix != 0 and ix != 826:
                Z = preds != torch.argmax(self.old_prediction, dim=1)
                W = self.old_gts == gts
                incons = (torch.sum(Z * W) / (416 * 416)).cpu().numpy()

                IoU['inconsistencies'].append(incons)

            self.old_prediction = prediction
            self.old_gts = gts

        return loss, prediction, IoU
