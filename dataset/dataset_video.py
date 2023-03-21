from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np
import glob
from utils.semantic_masks import binary_mask
import segmentation_models_pytorch as smp
from utils.albumentation import get_training_augmentation, get_preprocessing

class RAPNVideoDataset(BaseDataset):
    """RAPN100 Video Surgery Dataset. Read sequences of images, apply augmentation and preprocessing transformations.

    Args:
        video_dir (str): path to video folders
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
        valid (bool): True if dataset will be used as validation set
        sequence_length (int): number of images that compose a sequence for the convLSTM
        platform (str): 'server' or 'local' based on where you run the code

    """

    #CLASSES = ["Background", "Tool clasper", "Tool wrist", "Tool shaft", "Suturing needle", "Thread", "Suction tool",
     #      "Needle Holder", "Clamps", "Catheter"]
    #CLASSES = ["Background", "Instrument"]


    def __init__(
            self,
            video_dir,
            classes,
            augmentation=None,
            preprocessing=None,
            valid=False,
            sequence_length=5
    ):
        self.sequences = glob.glob(video_dir + '/raw_images/*/*')
        self.masks = glob.glob(video_dir + '/masks/*/*')
        self.sequences.sort()
        self.masks.sort()

        # convert str names to class values on masks
        self.classes = classes
        self.class_values = [classes.index(cls) for cls in classes]

        self.sequence_length = sequence_length
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        print(len(self.sequences), len(self.masks))
        # print(self.sequences)
        # print(self.masks)

    def __getitem__(self, i):
        # read data

        sequence = np.zeros((self.sequence_length, 3, 512, 512))
        #masks = np.zeros((self.sequence_length, len(self.classes), 512, 512))
        masks = np.zeros((self.sequence_length, 1, 512, 512))

        images_sequence = glob.glob(self.sequences[i] + '/*.png')
        masks_sequence = glob.glob(self.masks[i] + '/*.png')
        images_sequence.sort()
        masks_sequence.sort()
        images_sequence = images_sequence[-self.sequence_length:]
        masks_sequence = masks_sequence[-self.sequence_length:]

        # print(images_sequence)
        # print(masks_sequence)

        # get all the frames from a sequence with the corresponding masks
        for k in range(0, len(images_sequence)):

            image = cv2.imread(images_sequence[k])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(images_sequence[k], masks_sequence[k])
            mask = cv2.imread(masks_sequence[k], 0)  # 0 means grayscale
            # print(mask[400][400])
            if len(self.class_values) == 2:
                mask = binary_mask(mask)
            # mask = index_mask(mask, 512, 512)
            # print(mask[400][400])
            # print('image shape: ', image.shape)
            # print('mask shape: ', mask.shape)

            masks_ = [(mask == v) for v in self.class_values]
            mask = np.stack(masks_, axis=-1).astype('float')
            #print(mask.shape)

            # Augmentation
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # Preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)

            # print(sample['image'].shape, sample['mask'].shape)

            sequence[k] = sample['image']
            #masks[k] = sample['mask']
            masks[k] = np.array(sample['mask'][0] == 0)

        #print('lunghezze: ', sequence.shape, masks.shape)

        return sequence, masks

    def __len__(self):
        return len(self.sequences)


train_dir = '/Volumes/TOSHIBA EXT/Video_Segmentation_Dataset_RAPN'
classes = ['Background', 'Instrument']
preprocessing_fn = smp.encoders.get_preprocessing_fn('timm-mobilenetv3_large_100', 'imagenet')
train_dataset = RAPNVideoDataset(
        train_dir,
        classes=classes,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
sequence, masks = train_dataset.__getitem__(1)
print(len(train_dataset))
print(sequence.shape, masks.shape)

