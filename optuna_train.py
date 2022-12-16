"""
Script which performs an Optuna study to choose the best hyperparameters of the model specified in the
input parameters. Usually here the hyperparameters to evaluate are: Dropout, Activation Function, Loss Function,
Optimizer, Learning Rate
"""

import optuna
import os
import argparse

import numpy as np
import torch
import segmentation_models_pytorch as smp
from utils.albumentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
from dataset.dataset import RAPN_Dataset
from torch.utils.data import DataLoader
from epoch import TrainEpoch, ValidEpoch

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
    out_dir = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/RAPN_results/multiclass/models" + \
              f"/{MODEL_NAME}{ENCODER}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{LOSS}"
    train_dir = os.path.join(DATA_DIR, 'train')
    valid_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')
else:
    DATA_DIR = ""


def define_model(trial):

    # We optimize the dropout, the depth of the encoder and the activation function
    p = trial.suggest_float("dropout", 0.2, 0.5)
    #enc_depth = trial.suggest_int("encoder_depth", 3, 5)
    activation = trial.suggest_categorical("activation", ["sigmoid", "softmax2d", "tanh"])

    # create segmentation model with pretrained encoder
    if MODEL_NAME == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            #encoder_depth=enc_depth,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(classes),
            activation=activation,
            aux_params={'classes': len(classes), 'dropout': p}
        )
    elif MODEL_NAME == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            #encoder_depth=enc_depth,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            classes=len(classes),
            activation=activation,
            aux_params={'classes': len(classes), 'dropout': p}
        )
    elif MODEL_NAME == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            #encoder_depth=enc_depth,
            encoder_weights=ENCODER_WEIGHTS,
            decoder_channels=[256, 128, 64, 32, 16],
            classes=len(classes),
            activation=activation,
            aux_params={'classes': len(classes), 'dropout': p}
        )
    else:
        model = None

    return model


def objective(trial):

    # Generate the model
    model = define_model(trial).to(DEVICE)

    # Generate optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    #Choose the loss
    loss = trial.suggest_categorical("loss", ["focal", "dice", "focaldice"])
    if loss == 'focal':
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2)
        loss2 = None
    elif loss == 'dice':
        loss = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
        loss2 = None
    elif loss == 'focaldice':
        loss = smp.losses.FocalLoss(mode='multiclass', gamma=2)
        loss2= smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)
    else:
        loss = None
        loss2 = None

    # define preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)

    # TRAINING SET (I recommend using only a part of it)
    train_dataset = RAPN_Dataset(
        valid_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # VALIDATION SET
    valid_dataset = RAPN_Dataset(
        test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    # TRAIN AND VALIDATION LOADER
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)

    # SCHEDULER for the reduction of the learning rate when the learning stagnates
    # namely when the train loss doesn't decrease for a fixed amount of epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7)

    metrics = [
        # smp.utils.metrics.IoU(threshold=0.5),
    ]

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

    for i in range(1, NUM_EPOCHS + 1):

        print('\nEpoch: {}'.format(i))

        # Run a train epoch
        train_logs = train_epoch.run(train_loader)
        # Run a validation epoch
        valid_logs, valid_iou, valid_dice = valid_epoch.run(valid_loader)

        IoU = np.mean(valid_iou)

        trial.report(IoU, i)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        scheduler.step(train_logs['loss'])

    return IoU


def main():

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=226800)
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image('Optuna.png')

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("     Number of finished trials: ", len(study.trials))
    print("     Number of pruned trials: ", len(pruned_trials))
    print("     Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("     Value:", trial.value)

    print("     Params: ")
    for key, value in trial.params.items():
        print("     {}: {}".format(key, value))


if __name__ == '__main__':

    torch.cuda.empty_cache()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main()
