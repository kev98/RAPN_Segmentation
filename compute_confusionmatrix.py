import os
import argparse

import numpy as np
import torch
import pandas as pd
import segmentation_models_pytorch as smp
from utils.albumentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
from dataset.dataset import RAPN_Dataset
from torch.utils.data import DataLoader
from utils.helper_functions import saveResults, network_stats, create_dataframe, create_testdataframe
from utils.focalLoss import FocalLoss
from epoch import TrainEpoch, ValidEpoch
from utils.diceLoss import GDiceLoss, GDiceLossV2
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm as tqdm
from torchmetrics import ConfusionMatrix
import cv2
from utils.semantic_masks import class2color

# Parse input argument
parser = argparse.ArgumentParser(description="Network for segmentation in RAPN",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-b", "--batch", help="batch size", default=16)
parser.add_argument("-e", "--epochs", help="number of epochs", default=50)
parser.add_argument("-l", "--loss", help="loss function", default="focal")
parser.add_argument("-s", "--save", help="save criterion", default="mean_IoU")
parser.add_argument("-a", "--activation", help="activation function", default="softmax2d")
parser.add_argument("-p", "--profile", action="store_true", help="activation function")
parser.add_argument("platform", help="Platform on which the code is going to run")
parser.add_argument("encoder", help="Encoder of the network")
parser.add_argument("model", help="Architecture model of the network (FPN, DeepLab, U-Net...)")
args = parser.parse_args()
config = vars(args)

# Set the hyperparameters of the network
BATCH_SIZE = int(config['batch'])
NUM_EPOCHS = int(config['epochs'])
LOSS = config['loss']
SAVE_CRITERION = config['save']
ACTIVATION = config['activation']
PLATFORM = config['platform']
PROFILE = config['profile']
LEARNING_RATE = 1e-3
PATIENCE = 10

# Definition of the segmentation classes
#classes = ["Background", "Instrument"]
#classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
#           'Suction', 'Large Needle Driver', 'Echography']
classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
           'Vessel Loop', 'Cadiere Forceps', 'Gauze', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
           'Laparoscopic Fenestrated Forceps', 'Bulldog wire', 'Endobag', 'Veriset', 'Hemolock Clip Applier',
           'Laparoscopic Needle Driver', 'Other instruments']

#Choose the encoder and the segmentation model
ENCODER = config['encoder']  # encoder
ENCODER_WEIGHTS = 'imagenet'  # pretrained weights
CLASSES = classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = config['model']  # segmentation model
#classification_params = {'classes': 2, 'dropout': 0.3}

# DATA ROOT
if PLATFORM == "server":
    DATA_DIR = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN"
    out_dir = r"/home/kmarc/workspace/nas_private/RAPN_results/base_model/multiclass_all"
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
elif PLATFORM == "local":
    DATA_DIR = r"/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures"
    out_dir = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/RAPN_results/multiclass_1"
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
else:
    DATA_DIR = ""


def main():

    # create segmentation model with pretrained encoder
    if MODEL_NAME == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes':len(classes), 'dropout':0.38}
            # aux_params= classification_params
        )
    elif MODEL_NAME == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes':len(classes), 'dropout':0.38}
        )
    elif MODEL_NAME == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            decoder_channels=[256, 128, 64, 32, 16],
            classes=len(classes),
            activation=ACTIVATION
        )
    else:
        model = None

    # define preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)
    model.load_state_dict(torch.load("/home/kmarc/workspace/nas_private/RAPN_results/base_model/multiclass_all/FPNtu-efficientnetv2_rw_s_bs16_lr0.0003_focaldice_othclasses/tu-efficientnetv2_rw_s-FPN-30ce.pth"))
    model.to(DEVICE)
    model.eval()

    # VALIDATION SET
    valid_dataset = RAPN_Dataset(
        valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # TEST SET
    test_dataset = RAPN_Dataset(
        test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    #print(train_dataset.__getitem__(0)[1].shape)

    # TRAIN AND VALIDATION LOADER
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    confmat = ConfusionMatrix(task="multiclass", num_classes=len(classes))

    # CONFUSION MATRIX COMPUTATION ON THE VALIDATION SET
    total_confusion = np.zeros((len(classes), len(classes)))
    counter = np.zeros(len(classes))

    with tqdm(valid_loader, desc='valid', file=sys.stdout) as iterator:
        for x, y in iterator:

            prediction = model(x.to(DEVICE))
            
            if type(prediction) is tuple:
                prediction = prediction[0]

            prediction= prediction.detach().cpu()
            
            target = torch.argmax(y, dim=1)
            confusion = confmat(prediction, target).float().numpy()

            #divide each element for the sum of the pixels of that class in the image
            sum_conf = np.sum(confusion, axis=1)

            for idx, s in enumerate(sum_conf):
                if s != 0:
                    counter[idx] += 1
                    sum_conf = np.repeat(s, len(classes))
                    confusion[idx] = np.divide(confusion[idx], sum_conf)

            total_confusion = np.add(total_confusion, confusion)

            # Code to check if the confusion matrix is correctly computed
            '''frame = cv2.cvtColor(x.numpy().squeeze().transpose((1,2,0)), cv2.COLOR_BGR2RGB)

            prediction = np.argmax(prediction.detach().numpy().squeeze(), axis=0)
            fr = class2color(prediction)
            print(frame.shape, fr.shape)
            print(type(frame), type(fr))

            overlay = cv2.addWeighted(np.uint8(frame), 0.5, np.uint8(fr), 0.6, 1)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            # overlay = cv2.resize(overlay, orig_shape)

            cv2.imshow('window', overlay)

            cv2.waitKey(0)'''

    d = {}
    for idx, c in enumerate(counter):
        if c != 0:
            total_confusion[idx] = total_confusion[idx]/c
            d[classes[idx]] = list(total_confusion[idx])
        else:
            d[classes[idx]] = list(np.zeros(len(classes)))

    df = pd.DataFrame(d, index=classes, columns=classes)
    df = df.transpose()

    df.to_excel(out_dir + r'/confusion_matrix_validation.xlsx')

    # CONFUSION MATRIX COMPUTATION ON THE TEST SET
    total_confusion = np.zeros((len(classes), len(classes)))
    counter = np.zeros(len(classes))

    with tqdm(test_loader, desc='test', file=sys.stdout) as iterator:
        for x, y in iterator:

            prediction = model(x.to(DEVICE))

            if type(prediction) is tuple:
                prediction = prediction[0]

            prediction= prediction.detach().cpu()

            target = torch.argmax(y, dim=1)
            confusion = confmat(prediction, target).float().numpy()

            #divide each element for the sum of the pixels of that class in the image
            sum_conf = np.sum(confusion, axis=1)

            for idx, s in enumerate(sum_conf):
                if s != 0:
                    counter[idx] += 1
                    sum_conf = np.repeat(s, len(classes))
                    confusion[idx] = np.divide(confusion[idx], sum_conf)

            total_confusion = np.add(total_confusion, confusion)

    d = {}
    for idx, c in enumerate(counter):
        if c != 0:
            total_confusion[idx] = total_confusion[idx]/c
            d[classes[idx]] = list(total_confusion[idx])
        else:
            d[classes[idx]] = list(np.zeros(len(classes)))

    df = pd.DataFrame(d, index=classes, columns=classes)
    df = df.transpose()

    df.to_excel(out_dir + '/confusion_matrix_test.xlsx')


if __name__ == '__main__':

    torch.cuda.empty_cache()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main()
