import os
import pandas as pd
import numpy as np
from utils.albumentation_TC import get_training_augmentation, get_preprocessing
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from epoch_TC import TrainEpoch, ValidEpoch
from utils.focalLoss import FocalLoss
import convLSTM as clstm
import argparse
from dataset.dataset_video import RAPNVideoDataset
from utils.helper_functions_TC import saveResults, create_dataframe
import matplotlib.pyplot as plt

# Parse input argument
parser = argparse.ArgumentParser(description="Network for segmentation in RAPN",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-b", "--batch", help="batch size", default=1)
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
PATIENCE = 30

# Definition of the segmentation classes
classes = ["Background", "Instrument"]
# classes = ["Background", "Tool clasper", "Tool wrist", "Tool shaft", "Suturing needle", "Thread", "Suction tool",
 #     "Needle Holder", "Clamps", "Catheter"]

#  Choose the encoder and the segmentation model
ENCODER = config['encoder']  # encoder
ENCODER_WEIGHTS = 'imagenet'  # pretrained weights
CLASSES = classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = config['model']  # segmentation model
#classification_params = {'classes': 2, 'dropout': 0.3}

# DATA ROOT
if PLATFORM == "server":
    DATA_DIR = r"/home/kmarc/workspace/nas_private/Video_Segmentation_Dataset_RAPN2"
    out_dir = r"/home/kmarc/workspace/nas_private/RAPN_results_final/TC_model/convLSTM/binary" + f"/{MODEL_NAME}{ENCODER}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{LOSS}"
    #model_path = r"/home/kmarc/workspace/nas_private/binary/models_SARRARP50/FPNtu-efficientnetv2_rw_s_rd:5_bs16_lr0.001_focal/tu-efficientnetv2_rw_s-FPNce.pth"
    #model_path = r"/home/kmarc/workspace/nas_private/binary/models_ROBUSTMIS" + f"/{MODEL_NAME}{ENCODER}_bs8_lr0.001_focal/{ENCODER}-{MODEL_NAME}ce.pth"
    model_path = r"/home/kmarc/workspace/nas_private/RAPN_results_final/base_model/binary/FPNtu-efficientnetv2_rw_s_bs16_lr0.000225_focaldice/tu-efficientnetv2_rw_s-FPN-30ce.pth"
    train_dir = os.path.join(DATA_DIR, 'train') #'training' if RAPN or ROBUST; 'train' if SARRARP
    test_dir = os.path.join(DATA_DIR, 'test') #'testing' if RAPN or ROBUST; 'val' if SARRARP
elif PLATFORM == "local":
    DATA_DIR = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/DataSARRARP50"
    out_dir = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/multiclass/OutputSARRARP50_LSTM/models" + f"/{MODEL_NAME}{ENCODER}_dp_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{LOSS}"
    #model_path = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/OutputROBUSTMIS/Models/mobilenet_v2-FPNce.pth"
    model_path = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/multiclass/OutputSARRARP50/models" + f"/{MODEL_NAME}{ENCODER}_bs4_lr0.001_focal/{ENCODER}-{MODEL_NAME}ce.pth"
    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'val')
else: DATA_DIR = ""


class TCSegmentationNet(nn.Module):

    def __init__(self, model, feature_channels, feature_size, batch_size=1):
        super(TCSegmentationNet, self).__init__()
        # feature size
        self.feature_size = feature_size
        # base model
        self.segmentationModel = model
        # convolutional LSTM
        self.clstm = clstm.ConvLSTM(feature_channels, feature_channels, 3, 'tanh', DEVICE, torch.float32,
                                    'learn', batch_size, 1, 1,
                                    state_img_size=[feature_size, feature_size])
        # states, include hidden state and cell output
        self.states = None

        # old prediction
        self.old_prediction = None
        # old ground truth
        self.old_gt = None
        # correct pixels from old prediction
        self.old_prediction_correct = None
        # weight of temporal consistency loss
        self.v = 0

    def forward(self, x, y, elap, batch_size, reset_states, optimizer, compute_loss=True):
        focal = 0
        tc_loss = 0
        loss_fin = 0

        # if the states have to be reset (e.g. beginning of sequence)
        if reset_states:
            self.states = None

        # number of time steps of the sequence
        #print('input shape:', x.shape)
        time_steps = x.shape[1]
        for t in range(time_steps):
            # print('time_steps:', time_steps)
            a = 1

            # frame of current time step
            input = x[:, t, :, :]
            #print(input.shape)
            # extract features
            features = self.segmentationModel.encoder(input.float())
            # print('features:', len(features))
            # pass the features of the last layer if the encoder and the states to the ConvLSTM cell
            # output: the output [1, 272, 16, 16] of this step
            # self.states[0]: current cell output, sel.states[1]: current hidden state
            output, self.states = self.clstm.forward(features[-1], self.states, batch_size)
            #print('self.state:', self.states.shape)
            # print('output conv shape:', output.shape)

            # store the lower-level features (all except the last ones [1, 272, 16, 16] for efficientnetv2, 
            # in general [1, #channel_last_layer, width_last_layer, height_last_layer])
            feat = features[:-1]
            # append the features computed by the ConvLSTM cell
            feat.append(output)
            #print('decoder input shape:', features[-1].shape)
            # compute the decoder output
            #print(len(feat))
            decoder_output = self.segmentationModel.decoder(*feat)
            #print('decoder output shape:', decoder_output.shape)
            # compute the prediction
            prediction = self.segmentationModel.segmentation_head(decoder_output)
            #print('prediction shape', prediction.shape)
            #print('final prediction ', prediction)

            # if the states have not been reset and we want to compute the temporal consistency loss:
            if reset_states is False and compute_loss:
                A = self.old_gt == torch.argmax(y[:, t], dim=1)
                # check if almost one between the current frame prediction and the previous frame prediction is correct, pixel by pixel
                C = torch.bitwise_or(torch.argmax(prediction, dim=1) == torch.argmax(y[:, t], dim=1),
                                     self.old_prediction_correct)
                W = A * C
                W = W.unsqueeze(1)
                S = torch.nn.functional.one_hot(torch.argmax(y[:, t], dim=1).to(torch.int64), len(classes)).permute(0, 3, 1, 2)
                tc_loss = torch.nn.MSELoss(reduction='sum')(W * S * prediction, W * S * self.old_prediction) / (
                        torch.sum(A) + 1e-6)

            # if we want to compute the loss between the prediction at time t and the ground truth at time t
            if compute_loss:
                reset_states = False
                #focal = FocalLoss()(prediction, torch.argmax(y[:, t], dim=1))
                focal = smp.losses.FocalLoss(mode='binary', gamma=2.5)
                dice = smp.losses.DiceLoss(mode='binary', smooth=1e-5)
                total_loss = focal(prediction, torch.argmax(y[:, t], dim=1)) + dice(prediction, torch.argmax(y[:, t], dim=1))
                self.old_prediction = prediction.data
                self.old_gt = torch.argmax(y[:, t], dim=1).data
                self.old_prediction_correct = torch.argmax(prediction, dim=1) == self.old_gt

                # the final loss is the sum of the focal loss with the temporal consistency loss
                #loss_fin += a * focal + self.v * tc_loss
                loss_fin += a * total_loss + self.v * tc_loss

                if t == (time_steps - 1):
                    loss_fin.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        return prediction, focal, tc_loss

    def predict(self, x, y, elap, batch_size, reset_states, optimizer):

        if self.training:
            self.eval()

        with torch.no_grad():
            prediction, previous = self.forward(x, y)

        return prediction, previous


def main():

    # create segmentation model with pretrained encoder
    if MODEL_NAME == 'FPN':
        seg_model = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(classes),
            activation=ACTIVATION,
            # aux_params= classification_params
        )
    elif MODEL_NAME == 'DeepLabV3+':
        seg_model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            encoder_output_stride=16,
            classes=len(classes),
            activation=ACTIVATION
        )
    else:
        seg_model = None

    # preprocessing
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)

    # load state dict for the segmentation model
    # model_dir = r"/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/OutputRAPN/models/tu-effv2_bs4_lr0.001_focal"
    # model_path = glob.glob(model_dir + '/*.pth')[0]
    seg_model.load_state_dict(torch.load(model_path))
    seg_model.to(DEVICE)

    # build a temporal consistent segmentation Network with the previously defined segmentation model
    # feature_channels = 272 if we use efficientnetv2 as encoder, 360 for efficientnet_b4
    seg_model.encoder.eval()
    x = torch.rand((1, 3, 512, 512))
    features = seg_model.encoder(x.to(device=DEVICE))
    print('len features: ', len(features))
    print('shape last features: ', features[-1].shape)
    channels = features[-1].shape[1]
    model = TCSegmentationNet(seg_model, feature_size=len(classes), feature_channels=channels, batch_size=1)
    # model.load_state_dict(torch.load('CONVLSTM_effv2.pth'))
    model.to(DEVICE)

    train_dataset = RAPNVideoDataset(
        train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        valid=False,
        sequence_length=5
    )

    #a = train_dataset.__getitem__(0)
    #print(a[0].shape, a[1].shape)

    valid_dataset = RAPNVideoDataset(
        test_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        valid=True,
        sequence_length=5
    )

    # train and validation data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # loss function
    loss = FocalLoss()
    #loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.5)
    #loss2 = smp.losses.DiceLoss(mode='multiclass', smooth=1e-5)

    metrics = [
        # smp.utils.metrics.IoU(threshold=0.5),
    ]

    # training optimizer
    optimizer = torch.optim.AdamW([
        dict(params=model.parameters(), lr=LEARNING_RATE)
    ])

    # trainable layers
    names_trainable = [

        'segmentationModel.decoder.aspp.0.convs.0.0.weight', 'segmentationModel.decoder.aspp.0.convs.0.1.weight',
        'segmentationModel.decoder.aspp.0.convs.0.1.bias', 'segmentationModel.decoder.aspp.0.convs.1.0.0.weight',
        'segmentationModel.decoder.aspp.0.convs.1.0.1.weight', 'segmentationModel.decoder.aspp.0.convs.1.1.weight',
        'segmentationModel.decoder.aspp.0.convs.1.1.bias', 'segmentationModel.decoder.aspp.0.convs.2.0.0.weight',
        'segmentationModel.decoder.aspp.0.convs.2.0.1.weight', 'segmentationModel.decoder.aspp.0.convs.2.1.weight',
        'segmentationModel.decoder.aspp.0.convs.2.1.bias', 'segmentationModel.decoder.aspp.0.convs.3.0.0.weight',
        'segmentationModel.decoder.aspp.0.convs.3.0.1.weight', 'segmentationModel.decoder.aspp.0.convs.3.1.weight',
        'segmentationModel.decoder.aspp.0.convs.3.1.bias', 'segmentationModel.decoder.aspp.0.convs.4.1.weight',
        'segmentationModel.decoder.aspp.0.convs.4.2.weight', 'segmentationModel.decoder.aspp.0.convs.4.2.bias',
        'segmentationModel.decoder.aspp.0.project.0.weight', 'segmentationModel.decoder.aspp.0.project.1.weight',
        'segmentationModel.decoder.aspp.0.project.1.bias', 'segmentationModel.decoder.aspp.1.0.weight',
        'segmentationModel.decoder.aspp.1.1.weight', 'segmentationModel.decoder.aspp.2.weight',
        'segmentationModel.decoder.aspp.2.bias', 'segmentationModel.decoder.block1.0.weight',
        'segmentationModel.decoder.block1.1.weight', 'segmentationModel.decoder.block1.1.bias',
        'segmentationModel.decoder.block2.0.0.weight', 'segmentationModel.decoder.block2.0.1.weight',
        'segmentationModel.decoder.block2.1.weight', 'segmentationModel.decoder.block2.1.bias',
        'segmentationModel.segmentation_head.0.weight', 'segmentationModel.segmentation_head.0.bias',
        'clstm.h0', 'clstm.c0', 'clstm.cell.peephole_weights', 'clstm.cell.convolution.weight',
        'clstm.cell.convolution.bias',
        'clstm.cell.activation_function.weight', 'clstm1.h0', 'clstm1.c0', 'clstm1.cell.peephole_weights',
        'clstm1.cell.convolution.weight',
        'clstm1.cell.convolution.bias', 'clstm1.cell.activation_function.weight', 'clstm2.h0', 'clstm2.c0',
        'clstm2.cell.peephole_weights',
        'clstm2.cell.convolution.weight', 'clstm2.cell.convolution.bias', 'clstm2.cell.activation_function.weight',
        'segmentationModel.classification_head.1.weight', 'segmentationModel.classification_head.1.bias'
    ]

    # allow the update of some weights (convlstm, decoder and segmentation head) and block the encoder weights
    for name, parameter in model.named_parameters():
        if name not in names_trainable:
            parameter.requires_grad = False
        if len(name) > 18 and name[18] == 'd':
            parameter.requires_grad = True

        print(name, parameter.requires_grad)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7)

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        classes=CLASSES,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        classes=CLASSES,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0
    # results dataframe
    results_df = pd.DataFrame()

    if DEVICE == 'cuda':
        print(torch.cuda.memory_summary())

    for i in range(1, NUM_EPOCHS + 1):

        if i > 5:
            model.v = i / 10
        else:
            model.v = 0.2 #we could try to change it

        print('\nEpoch: {}'.format(i))
        # Train and validation epochs
        train_logs = train_epoch.run(train_loader, i)
        valid_logs, IoU, incons = valid_epoch.run(valid_loader, i)
        print("Results valid:")
        print(len(CLASSES), len(IoU))
        for c in range(len(CLASSES)):
            print(CLASSES[c], IoU[c])

        # save the performance of the epoch
        if i == NUM_EPOCHS:
            save = True
        else:
            save = False
        IoU, inference_time = saveResults(valid_loader, model, classes, PLATFORM, ENCODER, MODEL_NAME, save_img=save)
        curr_res = create_dataframe(model, i, IoU, inference_time, classes, train_logs, valid_logs, incons)
        results_df = pd.concat([results_df, curr_res])

        # if the IoU after the last epoch is better of the current max score, save the model
        if max_score < np.mean(IoU):
            max_score = np.mean(IoU)
            torch.save(model.state_dict(), out_dir + '/' + ENCODER + '-' + MODEL_NAME + 'ce' + '.pth')
            print('Model saved!')

        #results_df = results_df.append(
        #   {'IoU': IoU, 'Train Loss': train_logs['loss'],
        #     'Valid Loss': valid_logs['loss'], 'incons': incons, 'IoU': np.mean(IoU)}, ignore_index=True)
        #results_df.to_csv('convlstm_3.csv')
        print("Results \"save results\":")
        for c in range(len(CLASSES)):
            print(CLASSES[c], IoU[c])
            # writer.add_scalar('IoU/' + CLASSES[c][:5], IoU[c], i)
        print('Mean IoU: ', np.mean(IoU))

        scheduler.step(train_logs['loss'])

        # # TENSORBOARD POLOTS
        # writer.add_scalar('Loss/train', train_logs['loss'], i)
        # writer.add_scalar('Loss/valid', valid_logs['loss'], i)
        # writer.add_scalar('MeanIoU', np.mean(IoU), i)
        # writer.add_scalar('Cfactor', model.v, i)
        # writer.add_scalar('inconsistencies', incons, i)

        print('Max IoU: ', max_score)

        del train_logs, valid_logs, IoU, incons

    # Print and save plots of losses and mIoU
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="Valid and train loss")
    plt.plot(results_df['Valid Loss'], label='Valid Loss')
    plt.plot(results_df['Train Loss'], label='Train Loss')
    plt.legend(loc='best')
    '''ax0 = fig.add_subplot(121, title="Valid loss")
    plt.plot(results_df['Valid Loss'])'''
    ax1 = fig.add_subplot(122, title="mIoU")
    plt.plot(results_df['Mean IoU'])
    plt.savefig(os.path.join(out_dir, 'plots.png'))

    # Save excel with the training report
    results_df.to_excel(out_dir + '/' + ENCODER + '-' + MODEL_NAME + 'ce' + '.xlsx')

    # curr_res = create_dataframe(model, NUM_EPOCHS, IoU, inference_time, FBetaScore, classes, test_logs, DiceScore)
    # results_df = pd.concat([results_df, curr_res])


if __name__ == '__main__':
    torch.cuda.empty_cache()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    main()
