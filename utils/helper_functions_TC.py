import cv2
import numpy as np
from utils.semantic_masks import class2color
import torch
import time
import pandas as pd

background = (0, 0, 0)
monopolar_cs =  (255, 0, 0)
force_bipolar = (0, 0, 255)
suction = (0, 225, 0)
large_nd = (225, 225, 0)
prograsp_forceps = (0, 225, 255)
prograsp_forceps = (0, 0, 255)
fenestrated_bipolar = (255, 0, 255)
fenestrated_bipolar = (0, 0, 255)
echography = (255, 150, 0)


# segmentation_classes = [background, monopolar_cs, force_bipolar, suction, large_nd, prograsp_forceps, fenestrated_bipolar, echography]
VALID_SWITCH_INDEXES = [0, 230, 443, 573, 1151, 1155]

def computeIoU(predictions, targets, segmentation_classes):
    IoU = []
    for index in range(len(segmentation_classes)):
        targ = np.reshape(np.array(predictions) == index, (1, -1))
        pred = np.reshape(np.array(targets) == index, (1, -1))
        intersection = np.sum(np.bitwise_and(targ, pred))
        union = np.sum(np.bitwise_or(targ, pred))
        iou_class= (intersection/(union + 1e-10))
        IoU.append(iou_class)

    return IoU


def saveResults(X_test, model, segmentation_classes, platform, encoder, model_name, save_img=False):
    targets = []
    predictions = []
    comp_time = []
    ix = 0
    prev = torch.zeros(1, 4096, 32, 32)
    for images, masks, phase, elap, i, num_proc in X_test:
        targets.append(np.argmax(masks[:, -1], axis=1).cpu().numpy())
        #print('masks shape', masks.shape)
        #print((np.argmax(masks[:, -1], axis=1)).shape)


        if platform == 'server':
            images = images.to(device=torch.device('cuda'))


        with torch.no_grad():
            start = time.time()
            prediction, loss, tc_loss = model.forward(images, masks, elap, images.shape[0],
                                                           reset_states=True, optimizer=None,
                                                           compute_loss=False)
            res = np.argmax(prediction[0].cpu().numpy(), axis=0)
            #print(res)
            #print(prediction[0])
            #print(np.count_nonzero(res == 1))
            #print(np.count_nonzero(prediction[0][1]>0.5))
            predictions.append(res)
            end = time.time()
            comp_time.append(end - start)
            #print('predictions ', predictions)

            '''res = torch.argmax(prediction, dim=1)
            print('prediction',prediction)
            print('res', res)
            predictions.append(res)'''

        if save_img:
            mask = class2color(res)
            if platform == 'server':
                cv2.imwrite(r"/home/kmarc/workspace/nas_private/multiclass/TEST_SARRARP50_LSTM" + '/' + encoder + '-' + model_name + '_' + str(
                    ix) + '.png', mask[:, :, ::-1])
            elif platform == 'local':
                cv2.imwrite(
                    r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/OutputSARRARP50LSTM/TEST" + '/' + encoder + '-' + model_name + '_' + str(
                        ix) + '.png', mask[:, :, ::-1])
        ix = ix + 1

    IoU = computeIoU(predictions, targets, segmentation_classes)

    return IoU, np.mean(comp_time)


# function to create a dataframe on the valid epoch results
def create_dataframe(model, epoch, IoU, inference_time, classes, train_logs, valid_logs, incons):
    curr_res_dict = {}
    for j, v in enumerate(classes):
        dict_key = 'IoU_' + v.replace(' ', '')
        curr_res_dict[dict_key] = IoU[j]
    curr_res_dict['Mean IoU'] = np.mean(IoU)
    if incons is not None:
        curr_res_dict['Inconsistency'] = incons
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


n_classes = 2
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3


