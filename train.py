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
LEARNING_RATE = 2.45e-4
PATIENCE = 15

# Definition of the segmentation classes based on the model you want to train
# Binary 
#classes = ["Background", "Instrument"]
# Multiclass 1 (with or w/o 'Other instruments', with or w/o 'Cadiere Forceps')
#classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
#           'Suction', 'Large Needle Driver', 'Echography', 'Cadiere Forceps', 'Other instruments']
# Multiclass 2 (with or w/o 'Other instruments')
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
#           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
#           'Vessel Loop', 'Cadiere Forceps', 'Gauze', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
#           'Laparoscopic Fenestrated Forceps', 'Endobag', 'Veriset', 'Hemolock Clip Applier',
#           'Laparoscopic Needle Driver', 'Other instruments']
# Multiclass all classes
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
#           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
#           'Vessel Loop', 'Cadiere Forceps', 'Gauze', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
#           'Laparoscopic Fenestrated Forceps', 'Bulldog wire', 'Endobag', 'Veriset', 'Hemolock Clip Applier',
#           'Laparoscopic Needle Driver', 'Airseal trocar', 'Endobag wire', 'Endobag specimen retriever',
#           'Laparoscopic Clip Applier', 'Drain', 'Metal clip', 'Laparoscopic Scissors', 'Foam extruder',
#           'Assistant trocar', 'Fibrilar', 'Left PBP Needle Driver']
# Multiclass 14 classes of Francesco
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suction', 'Large Needle Driver', 'Suture wire',
#                    'Vessel Loop', 'Suture needle', 'Bulldog clamp', 'Echography', 'Laparoscopic Clip Applier', 'Gauze', 'Endobag']
# Multiclass 2 new (with 'Hemostasis') (with or w/0 'Other instruments)
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
#           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
#           'Vessel Loop', 'Cadiere Forceps', 'Hemostasis', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
#           'Laparoscopic Fenestrated Forceps', 'Endobag', 'Hemolock Clip Applier',
#           'Laparoscopic Needle Driver', 'Other instruments']
# Multiclass all classes new (with 'Hemostasis')
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
#           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
#           'Vessel Loop', 'Cadiere Forceps', 'Hemostasis', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
#           'Laparoscopic Fenestrated Forceps', 'Bulldog wire', 'Endobag', 'Hemolock Clip Applier',
#           'Laparoscopic Needle Driver', 'Airseal trocar', 'Endobag wire', 'Endobag specimen retriever',
#           'Laparoscopic Clip Applier', 'Drain', 'Foam', 'Metal clip', 'Surgical_Glove_Tip', 'Foam extruder',
#           'Laparoscopic Scissors', 'Assistant trocar']
# First experiment of merging (20 classes)
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
#           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle',
#           'Vessel Loop', 'Cadiere Forceps', 'Hemostasis', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
#           'Laparoscopic Fenestrated Forceps', 'Endobag', 'Hemolock Clip Applier', 'Laparoscopic Needle Driver',
#           'Other instruments']
# Second experiment of merging (19 classses)
classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle',
           'Vessel Loop', 'Cadiere Forceps', 'Hemostasis', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
           'Laparoscopic Fenestrated Forceps', 'Endobag', 'Hemolock Clip Applier', 'Other instruments']

#Choose the encoder and the segmentation model
ENCODER = config['encoder']  # encoder
ENCODER_WEIGHTS = 'imagenet'  # pretrained weights
CLASSES = classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = config['model']  # segmentation model
#classification_params = {'classes': 2, 'dropout': 0.3} # auxiliary parameters

# DATA ROOT
if PLATFORM == "server":
    DATA_DIR = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_tris"
    out_dir = r"/home/kmarc/workspace/nas_private/RAPN_results_final/base_model/multiclass_all" + \
              f"/{MODEL_NAME}{ENCODER}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{LOSS}_merge19_final"
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
            aux_params={'classes': len(classes), 'dropout': 0.386}
        )
    elif MODEL_NAME == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes': len(classes), 'dropout': 0.386}
        )
    elif MODEL_NAME == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            decoder_channels=[256, 128, 64, 32, 16],
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes': len(classes), 'dropout': 0.43}
        )
    else:
        model = None

    # define preprocessing function
    #preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)

    # Load an existing model
    #model.load_state_dict(torch.load("/home/kmarc/workspace/nas_private/RAPN_results/base_model/Francesco_model/tu-efficientnetv2_rw_s-FPN_14cl.pth"))
    #model.load_state_dict(torch.load("/home/kmarc/workspace/nas_private/RAPN_results_final/base_model/multiclass_all/DeepLabV3+tu-efficientnet_b4_bs8_lr0.0005_focaldice_merge19_final/tu-efficientnet_b4-DeepLabV3+-30ce.pth"))
    #model.load_state_dict(torch.load("/home/kmarc/workspace/nas_private/RAPN_results_final/base_model/multiclass_1/DeepLabV3+tu-efficientnet_b4_bs8_lr0.0003_focaldice/tu-efficientnet_b4-DeepLabV3+-30ce.pth"))

    model.to(DEVICE)

    # Get information about model
    if PROFILE:
        network_stats(model, DEVICE, BATCH_SIZE)

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

    # TEST SET
    test_dataset = RAPN_Dataset(
        test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    #print(train_dataset.__getitem__(0)[1].shape)

    # TRAIN, VALIDATION AND TEST LOADER
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # LOSSES (2 losses if you want to experience with a combination of losses, otherwise just pass None)
    if LOSS == 'focal':
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
        loss2 = None
    elif LOSS == 'dice':
        loss = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        loss2 = None
    elif LOSS == "focaldice":
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
        loss2 = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)

    metrics = [
        # smp.utils.metrics.IoU(threshold=0.5),
    ]

    # OPTIMIZER (here is set the starting learning rate)
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=LEARNING_RATE),
    ])

    # SCHEDULER for the reduction of the learning rate when the learning stagnates
    # namely when the valid loss doesn't decrease of a certain threshold for a fixed amount of epochs (patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-4,
                                                           factor=0.7, verbose=True)

    # create epoch runners
    # an epoch is a loop which iterates over dataloader`s samples
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

    # the validation epoch is used both for the validation and the test set
    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        loss2=loss2,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
        classes=classes
    )

    results_df = pd.DataFrame() # dataframe where the statistics of the training are written
    es_counter = 0 # Early-Stopping counter
    optimal_val_loss = 1000 # min value of the loss
    max_score = 0 # max value of the mean IoU

    # train model for N epochs
    for i in range(1, NUM_EPOCHS + 1):

        print('\nEpoch: {}'.format(i))

        # Run a train epoch
        train_logs = train_epoch.run(train_loader)
        # Run a validation epoch
        valid_logs, valid_iou, valid_dice = valid_epoch.run(valid_loader)

        # Every epoch (or every n epochs), compute the results and saves them into a dataframe
        if (i % 1) == 0:
            #save = True if i == NUM_EPOCHS else False # remove comment if you don't have a test set
            save = False
            # Compute the statistics of the epoch
            IoU, inference_time, FBetaScore, DiceScore = saveResults(valid_loader, model, len(classes), PLATFORM,
                                                                     ENCODER, MODEL_NAME, out_dir, save_img=save)
            # Create a Dataframe with the statistics
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

        # Stop the training if the PATIENCE is reached
        if es_counter >= PATIENCE:
            print(f"EARLY STOPPING... TRAINING IS STOPPED")
            break

        # Print IoU of each class
        for c in range(len(CLASSES)):
            print(CLASSES[c], IoU[c])

        scheduler.step(valid_logs['loss']) # a scheduler step based on the validation loss
        print('Ready for the next epoch')

        del train_logs, valid_logs, IoU

    # REMOVE COMMENT OF THE FOLLOWING 5 LINES IF YOU HAVE THE TEST SET
    # Run a test epoch
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
