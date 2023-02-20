# Some functions useful during the process

import cv2
import numpy as np
from utils.semantic_masks import class2color
import torch
import time
import torch.autograd.profiler as profiler
from torchsummaryX import summary
from torchmetrics.classification import MulticlassFBetaScore, BinaryFBetaScore
from torchmetrics import Dice
import pandas as pd
import os


def computeIoU(predictions, targets, num_classes):
    IoU = []
    for index in range(num_classes):
        targ = np.reshape(np.array(predictions) == index, (1, -1))
        pred = np.reshape(np.array(targets) == index, (1, -1))
        intersection = np.sum(np.bitwise_and(targ, pred))
        union = np.sum(np.bitwise_or(targ, pred))
        iou_class= (intersection/(union + 1e-10))
        IoU.append(iou_class)

    return IoU


def computeFBetaScore(predictions, targets, num_classes):

    if num_classes == 2:
        metric = BinaryFBetaScore(beta=1.0)
    else:
        metric = MulticlassFBetaScore(beta=1.0, num_classes=num_classes, average=None)

    return metric(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))


def computeDiceScore(predictions, targets):

    dice = Dice(average='micro')

    return dice(torch.tensor(np.array(predictions)), torch.tensor(np.array(targets)))


# function to compute metrics and save the masks segmented by the network
def saveResults(X_test, model, num_classes, platform, encoder, model_name, out_dir, save_img=False):
    ix = 0
    targets = []
    predictions = []
    comp_time = []
    for image, mask in X_test:
        targets.append(np.argmax(mask[0], axis=0).cpu().numpy())

        if platform == 'server':
            image = image.to(device=torch.device('cuda'))

        start = time.time()
        res = np.argmax(model.predict(image)[0].cpu().squeeze(), axis=0)
        predictions.append(res.numpy())
        end = time.time()

        # list of all inference times
        comp_time.append(end - start)

        # convert the mask to a colored mask and save it
        if save_img:
            save_dir = os.path.join(out_dir, 'TEST')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            mask = class2color(res, num_classes)
            cv2.imwrite(save_dir + '/predicted_mask_' + str(ix) + '.png', mask[:, :, ::-1])
        ix = ix + 1
    
    IoU = computeIoU(predictions, targets, num_classes)
    FBetaScore = np.array(computeFBetaScore(predictions, targets, num_classes))
    #DiceScore = np.array(computeDiceScore(predictions, targets))
    DiceScore = 0

    return IoU, sum(comp_time)/len(comp_time), FBetaScore, DiceScore


# function to print summary and profiling of the network
def network_stats(model, device, batch_size):
    # Print summary of the model layers given the input dimension
    print(summary(model.to(device), torch.rand(batch_size, 3, 512, 512).to(device)))
    input = torch.rand(batch_size, 3, 512, 512).to(device)
    '''
    # warm-up
    model(input)
    # profile a network forward
    cuda = True if device == 'cuda' else 'False'
    with profiler.profile(with_stack=True, profile_memory=True, use_cuda=cuda) as prof:
        out = model(input)
    if cuda:
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total'))
    else:
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))
    '''

# function to create a dataframe on the valid epoch results
def create_dataframe(model, epoch, IoU, inference_time, FBetaScore, classes, train_logs, valid_logs, DiceScore):
    curr_res_dict = {}
    for j, v in enumerate(classes):
        dict_key = 'IoU_' + v.replace(' ', '')
        curr_res_dict[dict_key] = IoU[j]
    curr_res_dict['Mean IoU'] = np.mean(IoU)
    if len(classes) > 2:
        for j, v in enumerate(classes):
            dict_key = 'FBetaScore_' + v.replace(' ', '')
            curr_res_dict[dict_key] = FBetaScore[j]
        curr_res_dict['Mean FBetaScore'] = np.mean(FBetaScore)
    else:
        curr_res_dict['Binary FBetaScore'] = FBetaScore
    curr_res_dict['Dice Score'] = DiceScore
    curr_res_dict['Mean Inference Time'] = inference_time
    curr_res_dict['fps'] = 1 / inference_time
    curr_res_dict['No of Parameters'] = sum(p.numel() for p in model.parameters())
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    curr_res_dict['Memory(MB)'] = '{:.3f}MB'.format(size_all_mb)
    curr_res_dict['Train Loss'] = train_logs['loss']
    curr_res_dict['Valid Loss'] = valid_logs['loss']

    return pd.DataFrame(curr_res_dict, index=[epoch])


# function to create a dataframe on the final test epoch results
def create_testdataframe(model, num_epochs, IoU, inference_time, FBetaScore, classes, test_logs, DiceScore):
    curr_res_dict = {}
    for j, v in enumerate(classes):
        dict_key = 'IoU_' + v.replace(' ', '')
        curr_res_dict[dict_key] = IoU[j]
    curr_res_dict['Mean IoU'] = np.mean(IoU)
    if len(classes) > 2:
        for j, v in enumerate(classes):
            dict_key = 'FBetaScore_' + v.replace(' ', '')
            curr_res_dict[dict_key] = FBetaScore[j]
        curr_res_dict['Mean FBetaScore'] = np.mean(FBetaScore)
    else:
        curr_res_dict['Binary FBetaScore'] = FBetaScore
    curr_res_dict['Dice Score'] = DiceScore
    curr_res_dict['Mean Inference Time'] = inference_time
    curr_res_dict['fps'] = 1 / inference_time
    curr_res_dict['No of Parameters'] = sum(p.numel() for p in model.parameters())
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    curr_res_dict['Memory(MB)'] = '{:.3f}MB'.format(size_all_mb)
    curr_res_dict['Test Loss'] = test_logs['loss']

    return pd.DataFrame(curr_res_dict, index=[num_epochs+1])


