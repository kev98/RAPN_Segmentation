import os
import argparse
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import numpy as np
import torch
import pandas as pd
import segmentation_models_pytorch as smp
from dataset import RAPN_Dataset
from torch.utils.data import DataLoader
from utils.albumentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
import tqdm

# Parse input argument
parser = argparse.ArgumentParser(description="Network for segmentation in RAPN",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--activation", help="activation function", default="softmax2d")
parser.add_argument("encoder", help="Encoder of the network")
parser.add_argument("model", help="Architecture model of the network (FPN, DeepLabV3+, U-Net...)")
args = parser.parse_args()
config = vars(args)

# Set the hyperparameters of the network
ACTIVATION = config['activation']
LEARNING_RATE = 3e-4
PATIENCE = 15

# Definition of the segmentation classes
#classes = ["Background", "Instrument"]
classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
           'Suction', 'Large Needle Driver', 'Echography']
iou_th = [0.7, 0.3, 0.3, 0, 0.3, 0.3, 0.3, 0.3]
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
#           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
#           'Vessel Loop', 'Cadiere Forceps', 'Gauze', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
#           'Laparoscopic Fenestrated Forceps', 'Bulldog wire', 'Endobag', 'Veriset', 'Hemolock Clip Applier',
#           'Laparoscopic Needle Driver']
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
#           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
#           'Vessel Loop', 'Cadiere Forceps', 'Gauze', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
#           'Laparoscopic Fenestrated Forceps', 'Bulldog wire', 'Endobag', 'Veriset', 'Hemolock Clip Applier',
#           'Laparoscopic Needle Driver', 'Other instruments']
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
#           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
#           'Vessel Loop', 'Cadiere Forceps', 'Gauze', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
#           'Laparoscopic Fenestrated Forceps', 'Bulldog wire', 'Endobag', 'Veriset', 'Hemolock Clip Applier',
#           'Laparoscopic Needle Driver', 'Airseal trocar', 'Endobag wire', 'Endobag specimen retriever',
#           'Laparoscopic Clip Applier', 'Drain', 'Metal clip', 'Laparoscopic Scissors', 'Foam extruder',
#           'Assistant trocar', 'Fibrilar', 'Left PBP Needle Driver']
#classes = ['Tissue', 'Monopolar Curved Scissors', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suction', 'Large Needle Driver', 'Suture wire',
#                    'Vessel Loop', 'Suture needle', 'Bulldog clamp', 'Echography', 'Laparoscopic Clip Applier', 'Gauze', 'Endobag']

#Choose the encoder and the segmentation model
ENCODER = config['encoder']  # encoder
ENCODER_WEIGHTS = 'imagenet'  # pretrained weights
CLASSES = classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = config['model']  # segmentation model

# DATA ROOT
DATA_DIR = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/train"
dest = "/home/kmarc/workspace/nas_private/sanity_check_output"
#DATA_DIR = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/test"
#dest = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures"


def computeIoU(prediction, target, num_classes):
    IoU = []
    for index in range(num_classes):
        targ = np.reshape(np.array(prediction) == index, (1, -1))
        pred = np.reshape(np.array(target) == index, (1, -1))
        intersection = np.sum(np.bitwise_and(targ, pred))
        union = np.sum(np.bitwise_or(targ, pred))
        iou_class= (intersection/(union + 1e-10))
        IoU.append(iou_class)

    return IoU


def main():

    # create segmentation model with pretrained encoder
    if MODEL_NAME == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes': len(classes), 'dropout': 0.38}
        )
    elif MODEL_NAME == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes': len(classes), 'dropout': 0.38}
        )
    elif MODEL_NAME == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            decoder_channels=[256, 128, 64, 32, 16],
            classes=len(classes),
            activation=ACTIVATION,
            aux_params={'classes': len(classes), 'dropout': 0.38}
        )
    else:
        model = None

    # define preprocessing function
    #preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)
    model.load_state_dict(torch.load("/home/kmarc/workspace/nas_private/RAPN_results/base_model/multiclass_1/DeepLabV3+tu-efficientnet_b4_bs8_lr0.0007_focal/tu-efficientnet_b4-DeepLabV3+-30ce.pth"))
    #model.load_state_dict(torch.load("/Users/kevinmarchesini/Desktop/tu-efficientnet_b4-DeepLabV3+-30ce.pth", map_location='cpu'))
    model.to(DEVICE)

    # TEST SET
    test_dataset = RAPN_Dataset(
        DATA_DIR,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    #print(train_dataset.__getitem__(0)[1].shape)

    dir_proc = os.listdir(os.path.join(DATA_DIR, 'raw_images'))
    dict_proc = {}
    #print(dir_proc)
    for d in dir_proc:
        if '._' not in d:
            dict_proc[d] = []

    print(dict_proc)

    #for image, mask, p, num in test_loader:
    for sample in tqdm.tqdm(test_loader):
        image, mask, p, num = sample
        p, num = p[0], num[0]
        mask = np.argmax(mask[0], axis=0).cpu().numpy()

        if DEVICE == 'cuda':
            image = image.to(device=torch.device('cuda'))

        pred = np.argmax(model.predict(image)[0].cpu().squeeze(), axis=0)

        IoU = computeIoU(pred, mask, len(classes))
        #print(p, num, IoU)

        # if the IoU of a class present in the real mask is low than the threshold we add the image to the list
        str = ""
        for i in range(len(classes)):
            if np.any(mask == i):
                if IoU[i] < iou_th[i]:
                    str = str + " " + classes[i]
        if str != "":
            dict_proc[p].append(num + str)

    for k in dict_proc.keys():
        dict_proc[k] = pd.Series(dict_proc[k])

    # create a Dataframe with the occurencies of each class
    df = pd.DataFrame.from_dict(dict_proc)
    #print(df)

    df.to_excel(dest + '/check_8classes_train.xlsx')


if __name__ == '__main__':

    torch.cuda.empty_cache()
    main()
