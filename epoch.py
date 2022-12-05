import sys

import numpy as np
import torch
from tqdm import tqdm as tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.meter import AverageValueMeter
import torch.autograd.profiler as profiler


# from focalLoss import FocalLoss

class Epoch:

    def __init__(self, model, loss, loss2, metrics, stage_name, classes, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.loss2 = loss2
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

    def batch_update(self, x, y, Iou):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        IoU = {'intersection': [0] * self.num_classes, 'union': [0] * self.num_classes}

        # show progress bar with tqdm
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred, IoU = self.batch_update(x, y, IoU)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()

                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs (potentially useless since metrics is empty)
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
                valid_DICE = np.divide([sum(x) for x in zip(IoU['intersection'], IoU['intersection'])],
                                       [sum(x) for x in zip(IoU['intersection'], IoU['union'])])

            else:
                return logs

        return logs, valid_IoU, valid_DICE


class TrainEpoch(Epoch):

    def __init__(self, model, loss, loss2, metrics, optimizer, classes, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            loss2=loss2,
            metrics=metrics,
            stage_name='train',
            classes=classes,
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, IoU):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        # + self.loss2(prediction, y)

        if self.loss2 is None:
            loss = self.loss(prediction, torch.argmax(y, dim=1))
        else:
            loss = self.loss(prediction, torch.argmax(y, dim=1)) + self.loss2(prediction, torch.argmax(y, dim=1))

        loss.backward()
        self.optimizer.step()
        return loss, prediction, None


class ValidEpoch(Epoch):

    def __init__(self, model, loss, loss2, metrics, classes, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            loss2=loss2,
            metrics=metrics,
            stage_name='valid',
            classes=classes,
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, IoU):
        with torch.no_grad():
            prediction = self.model.forward(x)
            # + self.loss2(prediction, y)

            if self.loss2 is None:
                loss = self.loss(prediction, torch.argmax(y, dim=1))
            else:
                loss = self.loss(prediction, torch.argmax(y, dim=1)) + self.loss2(prediction, torch.argmax(y, dim=1))

            preds = torch.argmax(prediction, dim=1)
            gts = torch.argmax(y, dim=1)

            i = []
            u = []
            for e in range(self.num_classes):
                A = preds == e
                B = gts == e
                i.append(torch.bitwise_and(A, B).int().sum().cpu().numpy().min())
                u.append(torch.bitwise_or(A, B).int().sum().cpu().numpy().min())

            IoU['intersection'] = [sum(x) for x in zip(IoU['intersection'], i)]
            IoU['union'] = [sum(x) for x in zip(IoU['union'], u)]
        #    loss = self.loss(prediction, y)
        return loss, prediction, IoU
