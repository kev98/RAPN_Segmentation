import numpy as np
import torch
import segmentation_models_pytorch as smp
from utils.albumentation import get_inference_preprocessing
import cv2
import glob


#Directory to explore probably in a recursive way to take the input images
#in_dir = "/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN"
in_dir = "/Volumes/ORSI/Kevin/video_dataset"

# Definition of the segmentation classes
#classes = ["Background", "Instrument"]
classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
           'Suction', 'Large Needle Driver', 'Echography']

#  Choose the encoder and the segmentation model
ENCODER = 'tu-efficientnet_b4'  # encoder
ENCODER_WEIGHTS = 'imagenet'  # pretrained weights
CLASSES = classes
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'DeepLabV3+'  # segmentation model
ACTIVATION = 'softmax2d'
#classification_params = {'classes': 2, 'dropout': 0.3}


def main():

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        encoder_output_stride=16,
        classes=len(classes),
        activation=ACTIVATION,
        aux_params={'classes': len(classes), 'dropout': 0.38}
    )

    # define preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', ENCODER_WEIGHTS)

    # load state dict for the segmentation model
    #model_path = r"/home/kmarc/workspace/nas_private/binary/models_pseudolabel/DeepLabV3+tu-efficientnet_b4_bs8_lr0.001_focal/tu-efficientnet_b4-DeepLabV3+ce.pth"
    model_path = r"/Users/kevinmarchesini/Desktop/tu-efficientnet_b4-DeepLabV3+-30ce.pth"

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(DEVICE)
    model.eval()

    images = glob.glob(in_dir + '/raw_images/*/*/*')
    print(len(images))
    # invece che rimuoverla devi cambiare la maschera e risalvarla con lo stesso nome (però modificarla per renderla uniforme alle altre)
    images = [x for x in images if "RAPN" not in x.split('/')[-1]]
    print(len(images))
    #loop on in_dir and construct list of images to label

    for image in images:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pre-processing to use the image as input of the NN
        sample = get_inference_preprocessing(preprocessing_fn)(image=img)
        img = sample['image']
        print(img.shape)
        img = np.reshape(img, (1, 3, 512, 512))
        img = torch.from_numpy(img)

        with torch.no_grad():
            prediction = model.forward(img.to(DEVICE))[0]
            prediction = torch.squeeze(prediction, 0)
            #prediction = np.random.randint(0,10,(2,3,3))
            #print('prediction: ', torch.squeeze(prediction, 0))
            mask = np.argmax(prediction.cpu().numpy(), axis=0)
            #print(np.count_nonzero(mask))

        #saves the mask in the correct folder and with the correct name
        out_dir = image.replace('raw_images', 'masks')
        print(out_dir)
        cv2.imwrite(out_dir, mask[:, :])


if __name__ == '__main__':

    #torch.cuda.empty_cache()
    '''if not os.path.exists(out_dir):
        os.mkdir(out_dir)'''
    main()
