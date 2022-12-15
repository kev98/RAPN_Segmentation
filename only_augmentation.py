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
classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
           'Suction', 'Large Needle Driver', 'Echography']

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
    out_dir = r"/home/kmarc/workspace/nas_private/RAPN_results/base_model/multiclass_1" + \
              f"/{MODEL_NAME}{ENCODER}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{LOSS}"
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
elif PLATFORM == "local":
    DATA_DIR = r"/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures"
    out_dir = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/RAPN_results/multiclass_1/models" + \
              f"/{MODEL_NAME}{ENCODER}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{LOSS}"
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
else:
    DATA_DIR = ""


def main():

    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)


    # TRAINING SET
    train_dataset = RAPN_Dataset(
        train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # VALIDATION SET
    valid_dataset = RAPN_Dataset(
        valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # TRAINING SET
    test_dataset = RAPN_Dataset(
        test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # train model for N epochs
    results_df = pd.DataFrame()
    es_counter = 0
    optimal_val_loss = 1000
    max_score = 0

    for i in range(1, NUM_EPOCHS + 1):

        print('\nEpoch: {}'.format(i))

        for k in range(len(train_dataset)):
            a = train_dataset.__getitem__(k)[1]

        for k in range(len(valid_dataset)):
            a = valid_dataset.__getitem__(k)[1]

        for k in range(len(test_dataset)):
            a = test_dataset.__getitem__(k)[1]


if __name__ == '__main__':

    torch.cuda.empty_cache()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main()
