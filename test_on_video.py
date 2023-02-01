import os

import cv2
import segmentation_models_pytorch as smp
import torch
from alive_progress import alive_bar

from utils.semantic_masks import class2color
#from crop_img import crop
import time
import numpy as np

from utils.albumentation import get_preprocessing

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# If you use the binary model
classes = ['background', 'instrument']
# If you use the multiclass model
#classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
#           'Suction', 'Large Needle Driver', 'Echography']
model_path = r'/home/kmarc/workspace/nas_private/RAPN_results/base_model/binary/FPNtu-efficientnetv2_rw_s_bs16_lr0.0002_focaldice/tu-efficientnetv2_rw_s-FPN-30ce.pth'

#Choose the encoder and the segmentation model
ENCODER = 'tu-efficientnetv2_rw_s'  # encoder
ENCODER_WEIGHTS = 'imagenet'  # pretrained weights
ACTIVATION = 'softmax2d'
CLASSES = classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'FPN'  # segmentation model


# create segmentation architecture
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

model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
model.to(DEVICE).eval()

IMAGE_SIZE = 512
preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)
preprocessing = get_preprocessing(preprocessing_fn)

#video_path = '/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/test_on_video/201807030911_48_4_AM_short.mp4'
video_path = '/home/kmarc/workspace/nas_private/inference_video/cut1_01.mp4' # path of the inference video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out_folder = '/Users/kevinmarchesini/Desktop/Internship @ Orsi Academy/test_on_video/test_video_out'
out_folder = '/home/kmarc/workspace/nas_private/inference_video/video_segmented_mymodel' # where the video segmented will be saved
out_path = out_folder + '/' + os.path.basename(video_path).replace('.mp4', '_segm.mp4')
out = cv2.VideoWriter(out_path, fourcc, fps, (512, 512))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = -1
scores = np.zeros((512, 512, 5), dtype=int)

with alive_bar(num_frames) as bar:
    while True: # change condition if you want not to segment every frame
        frame_number += 1
        ret, frame = cap.read()
        if not ret:
            break
        if ((frame_number % 1) == 0):
            start = time.time()
            #frame = crop(frame)

            #frame = cv2.resize(frame, (512, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inference of the model
            image = torch.tensor(preprocessing(image=frame)['image']).unsqueeze(0).to(DEVICE)
            #result = torch.bitwise_not(model.predict(image)[0] > 0.5).float().cpu().detach().numpy()[0] # if use Jente's moodel
            result = np.argmax(model.predict(image)[0].cpu().squeeze(), axis=0) # if use my model
            # Convert the masked in a colored masks
            fr = class2color(result)

            # Overlay of the colored mask on the original video
            overlay = cv2.addWeighted(np.uint8(frame), 0.5, np.uint8(fr), 0.6, 1)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            #overlay = cv2.resize(overlay, orig_shape)

            #cv2.imshow('window', overlay)
            end = time.time()
            out.write(overlay)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        bar()

cap.release()
out.release()
#cv2.destroyAllWindows()