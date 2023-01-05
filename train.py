import os
import argparse

import numpy as np
import torch
import pandas as pd
import segmentation_models_pytorch as smp
from dataset.dataset import RAPN_Dataset
from torch.utils.data import DataLoader
from utils.albumentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils.helper_functions import saveResults, network_stats, create_dataframe, create_testdataframe
from epoch import TrainEpoch, ValidEpoch
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
LEARNING_RATE = 3e-4
PATIENCE = 15

# Definition of the segmentation classes
classes = ["Background", "Instrument"]
#classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
#           'Suction', 'Large Needle Driver', 'Echography']

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
    out_dir = r"/home/kmarc/workspace/nas_private/RAPN_results/base_model/binary" + \
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

    # create segmentation model with pretrained encoder
    if MODEL_NAME == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes':2, 'dropout':0.45}
        )
    elif MODEL_NAME == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes':2, 'dropout':0.45}
        )
    elif MODEL_NAME == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            decoder_channels=[256, 128, 64, 32, 16],
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes':2, 'dropout':0.45}
        )
    else:
        model = None

    # define preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)
    #model.load_state_dict(torch.load("/home/kmarc/workspace/nas_private/RAPN_results/base_model/multiclass_1/UNet++tu-efficientnetv2_rw_s_bs16_lr0.001_focal/tu-efficientnetv2_rw_s-UNet++-30ce.pth"))
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

    #print(train_dataset.__getitem__(0)[1].shape)

    # TRAIN AND VALIDATION LOADER
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # LOSSES (2 losses if you want to experience with a combination of losses, otherwise just pass None)
    if LOSS == 'focal':
        #loss = FocalLoss() implementation of loss previously used by Francesco's
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
        loss2 = None
    elif LOSS == 'dice':
        #loss = GDiceLoss() implementation of loss previously used by Francesco's
        loss = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        loss2 = None
    elif LOSS == "focaldice":
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
        loss2 = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        #loss = FocalLoss() implementation of loss previously used by Francesco's
        #loss2 = GDiceLossV2() implementation of loss previously used by Francesco's

    metrics = [
        # smp.utils.metrics.IoU(threshold=0.5),
    ]

    # OPTIMIZER (you can set here the starting learning rate)
    optimizer = torch.optim.RMSprop([
        dict(params=model.parameters(), lr=LEARNING_RATE),
    ])

    # SCHEDULER for the reduction of the learning rate when the learning stagnates
    # namely when the valid loss doesn't decrease for a fixed amount of epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=2e-4,
                                                           factor=0.3, verbose=True)

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

    results_df = pd.DataFrame()
    es_counter = 0
    optimal_val_loss = 1000
    max_score = 0

    # train model for N epochs
    for i in range(1, NUM_EPOCHS + 1):

        print('\nEpoch: {}'.format(i))

        # Run a train epoch
        train_logs = train_epoch.run(train_loader)
        # Run a validation epoch
        valid_logs, valid_iou, valid_dice = valid_epoch.run(valid_loader)

        # Every epoch, compute the results and saves them into a dataframe
        if (i % 1) == 0:
            #save = True if i == NUM_EPOCHS else False
            save = False
            IoU, inference_time, FBetaScore, DiceScore = saveResults(valid_loader, model, len(classes), PLATFORM,
                                                                     ENCODER, MODEL_NAME, out_dir, save_img=save)
            curr_res = create_dataframe(model, i, IoU, inference_time, FBetaScore, classes,
                                        train_logs, valid_logs, DiceScore)
            results_df = pd.concat([results_df, curr_res])
            results_df.to_excel(out_dir + '/' + ENCODER + '-' + MODEL_NAME + f'-{NUM_EPOCHS}' + 'ce' + '.xlsx')
        else:
            IoU = valid_iou

        # update the model if the loss is decreased
        if SAVE_CRITERION == "validation_loss":
            if valid_logs['loss'] < optimal_val_loss:
                optimal_val_loss = valid_logs['loss']
                # optimal_val_acc = np.mean(IoU)
                # optimal_epoch = i+1
                es_counter = 0
                torch.save(model.state_dict(), out_dir + '/' + ENCODER + '-' + MODEL_NAME +
                           f'-{NUM_EPOCHS}' + 'ce' + '.pth')
                print('Model saved!')
            else:
                es_counter += 1
                print(f"EarlyStopping-counter = {es_counter}")
        # update the model if the mean IoU is increased
        elif SAVE_CRITERION == "mean_IoU":
            curr_mean_IoU = np.mean(IoU)
            print('Current mean IoU: ', curr_mean_IoU)
            if curr_mean_IoU > max_score:
                max_score = curr_mean_IoU
                es_counter = 0
                torch.save(model.state_dict(), out_dir + '/' + ENCODER + '-' + MODEL_NAME +
                           f'-{NUM_EPOCHS}' + 'ce' + '.pth')
                print('Model saved!')
            else:
                es_counter += 1
                print(f"EarlyStopping-counter = {es_counter}")

        if es_counter >= PATIENCE:
            print(f"EARLY STOPPING... TRAINING IS STOPPED")
            break

        for c in range(len(CLASSES)):
            print(CLASSES[c], IoU[c])

        scheduler.step(valid_logs['loss'])
        print('Ready for the next epoch')

        del train_logs, valid_logs, IoU

    # REMOVE COMMENTOF THE FOLLOWING 5 LINES IF YOU HAVE THE TEST SET
    test_logs, test_iou, test_dice = valid_epoch.run(test_loader)
    IoU, inference_time, FBetaScore, DiceScore = saveResults(test_loader, model, len(classes), PLATFORM, ENCODER,
                                                             MODEL_NAME, out_dir, save_img=True)
    curr_res = create_testdataframe(model, NUM_EPOCHS, IoU, inference_time, FBetaScore, classes, test_logs, DiceScore)
    results_df = pd.concat([results_df, curr_res])

    # Print and save plots of losses and mIoU
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="Valid and train loss")
    plt.plot(results_df['Valid Loss'], label='Valid Loss')
    plt.plot(results_df['Train Loss'], label='Train Loss')
    plt.legend(loc='best')
    ax1 = fig.add_subplot(122, title="mIoU")
    plt.plot(results_df['Mean IoU'])
    plt.savefig(os.path.join(out_dir, f'plots-{NUM_EPOCHS}.png'))

    # Save excel with the training report
    results_df.to_excel(out_dir + '/' + ENCODER + '-' + MODEL_NAME + f'-{NUM_EPOCHS}' + 'ce' + '.xlsx')


if __name__ == '__main__':

    torch.cuda.empty_cache()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main()
