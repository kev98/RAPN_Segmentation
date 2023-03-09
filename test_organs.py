import os
import argparse

import numpy as np
import torch
import pandas as pd
import segmentation_models_pytorch as smp
from dataset.dataset_organs import DSAD_Dataset_Binary
from torch.utils.data import DataLoader
from utils.albumentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils.helper_functions import saveResults, network_stats, create_dataframe, create_testdataframe
from epoch import TrainEpoch, ValidEpoch
import matplotlib.pyplot as plt
from sem_mask import class2color

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
LEARNING_RATE = 1.738e-4
PATIENCE = 15

# Definition of the segmentation classes
classes = ['other', 'abdominal_wall']
main_class = classes[1]

#Choose the encoder and the segmentation model
ENCODER = config['encoder']  # encoder
ENCODER_WEIGHTS = 'imagenet'  # pretrained weights
CLASSES = classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = config['model']  # segmentation model
#classification_params = {'classes': 2, 'dropout': 0.3}

# DATA ROOT
if PLATFORM == "server":
    DATA_DIR = r"/home/kmarc/workspace/nas_private/DSAD_Dataset/" + main_class
    out_dir = r"/home/kmarc/workspace/nas_private/DSAD_results_binary" + \
              f"/{main_class}/{MODEL_NAME}{ENCODER}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{LOSS}_inference"
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'test')
elif PLATFORM == "local":
    DATA_DIR = r"/Volumes/TOSHIBA EXT/DSAD_Dataset/" + main_class
    out_dir = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/DSAD_results_binary" + \
              f"/{main_class}/{MODEL_NAME}{ENCODER}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{LOSS}"
    train_dir = os.path.join(DATA_DIR, '')
    valid_dir = os.path.join(DATA_DIR, '')
else:
    DATA_DIR = ""


def main():

    # create segmentation model with pretrained encoder
    if MODEL_NAME == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            decoder_dropout=0.3173,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(classes),
            activation=ACTIVATION,
            #aux_params={'classes': len(classes), 'dropout': 0.38}
        )
    elif MODEL_NAME == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            classes=len(classes),
            activation=ACTIVATION,
            #aux_params={'classes': len(classes), 'dropout': 0.38}
        )
    elif MODEL_NAME == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            decoder_channels=[256, 128, 64, 32, 16],
            classes=len(classes),
            activation=ACTIVATION,
            #aux_params={'classes': len(classes), 'dropout': 0.38}
        )
    else:
        model = None

    # define preprocessing function
    #preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)
    model.load_state_dict(torch.load("/home/kmarc/workspace/nas_private/DSAD_results_binary/abdominal_wall/FPNtu-efficientnetv2_rw_s_bs16_lr0.00023421_focaldice/tu-efficientnetv2_rw_s-FPN-30ce.pth"))
    model.to(DEVICE)

    # Get information about model
    if PROFILE:
        network_stats(model, DEVICE, BATCH_SIZE)

    # Get information about model complexity: number of parameters and multiply - accumulate
    # operations per second
    # macs, params = get_model_complexity_info(model, (1, 3, 512, 512), as_strings=True,
                                            # print_per_layer_stat=False, verbose=True)
    # print(macs, params)

    # TRAINING SET
    train_dataset = DSAD_Dataset_Binary(
        train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # VALIDATION AND TEST SET
    valid_dataset = DSAD_Dataset_Binary(
        valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    #print(train_dataset.__getitem__(0)[1].shape)

    # TRAIN AND VALIDATION LOADER
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # LOSSES (2 losses if you want to experience with a combination of losses, otherwise just pass None)
    if LOSS == 'focal':
        #loss = FocalLoss() implementation of loss previously used by Francesco's
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2)
        loss2 = None
    elif LOSS == 'dice':
        #loss = GDiceLoss() implementation of loss previously used by Francesco's
        loss = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        loss2 = None
    elif LOSS == "focaldice":
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2)
        loss2 = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        #loss = FocalLoss() implementation of loss previously used by Francesco's
        #loss2 = GDiceLossV2() implementation of loss previously used by Francesco's

    metrics = [
        # smp.utils.metrics.IoU(threshold=0.5),
    ]

    # OPTIMIZER (you can set here the starting learning rate)
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=LEARNING_RATE),
    ])

    # SCHEDULER for the reduction of the learning rate when the learning stagnates
    # namely when the valid loss doesn't decrease for a fixed amount of epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=5e-4,
                                                           factor=0.7, verbose=True)

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = TrainEpoch(
        model,
        loss=loss,
        loss2=loss2,
        metrics=metrics,
        optimizer=optimizer,
        classes=classes,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        loss2=loss2,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
        classes=classes
    )

    ix = 0
    targets = []
    predictions = []
    comp_time = []
    for image, mask in valid_loader:
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
        save_dir = os.path.join(out_dir, 'TEST')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        mask = class2color(res)
        cv2.imwrite(save_dir + '/predicted_mask_' + str(ix) + '.png', mask[:, :, ::-1])
        ix = ix + 1


if __name__ == '__main__':

    torch.cuda.empty_cache()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main()
