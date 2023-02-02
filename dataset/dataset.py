import glob
from torch.utils.data import Dataset as BaseDataset
import cv2
# to avoid multiprocessing deadlocks
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
from utils.semantic_masks import binary_mask
import json


class RAPN_Dataset(BaseDataset):
    """RAPN100 Dataset. Read images and corresponding masks, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline (e.g. flip, scale etc.)
        preprocessing (albumentations.Compose): data preprocessing (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(
            self,
            images_dir,
            classes,
            augmentation=None,
            preprocessing=None,
    ):

        self.images = glob.glob(images_dir + '/raw_images/*/*.png')
        self.masks = glob.glob(images_dir + '/masks/*/*.png')
        self.images.sort()
        self.masks.sort()
        print('number of images: ',len(self.images))
        print('number of masks: ', len(self.masks))

        # convert str names to class values on masks, using the "config.json" file
        self.classes = classes
        with open("config_files/config.json", 'r') as openfile:
            # Reading from json file
            json_file = json.load(openfile)
        mapping = json_file["labels"]
        self.class_values = []
        self.other_label = json_file['classes'] + 1
        if len(classes) == 2:
            self.class_values.append(0)
            self.class_values.append(1)
        else:
            for cls in classes:
                if cls == 'Other instruments':
                    self.class_values.append(self.other_label)
                    continue
                if cls == 'Tissue' or cls == 'Background':
                    self.class_values.append(0)
                    continue
                if cls not in mapping.keys():
                    cls = cls + '_1'
                if cls not in mapping.keys():
                    cls = cls.replace('_1', '_01')
                try:
                    self.class_values.append(mapping[cls]["label"])
                except KeyError as e:
                    print('I got a KeyError - Class not valid: %s'%str(e))
        print('class values: ', self.class_values)

        # List of instruments to be annotated as "Other instruments" (not always necessary)
        self.other_instruments = []
        # N.B. in any case "Other instruments" class must be the last one of the self.classes list
        if self.classes[-1] == 'Other instruments':
            for i in range(1, self.other_label):
                if i not in self.class_values:
                    self.other_instruments.append(i)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read image and mask
        #print('image: ', self.images[i])
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], 0)

        # if we need the class "Other instruments"
        # N.B. in any case "Other instruments" class must be the last one of the self.classes list
        if self.classes[-1] == 'Other instruments':
            for oth in self.other_instruments:
                mask[mask == oth] = self.other_label

        # if we want to create a dataset for binary segmentation
        if len(self.classes) == 2:
            # set the "Inside body" class as background
            mask[mask == 15] = 0
            mask = binary_mask(mask)
            # separate the masks of the different classes and stack them
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
        # if we want to create a dataset for multiclass segmentation
        else:
            # separate the masks of the different classes and stack them
            mask[mask == 2] = 31 #conversion of the Bulldog wire to Suture wire
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
            sum = np.sum(mask, axis=2)
            result = np.logical_not(np.logical_xor(sum, mask[:, :, 0])).astype('float')
            mask[:, :, 0] = result

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images)


# Code to try the dataloader

'''classes = ['Background', 'Instrument']
classes = ['Tissue', 'Force Bipolar', 'Fenestrated Bipolar Forceps', 'Prograsp Forceps', 'Monopolar Curved Scissors',
           'Suction', 'Large Needle Driver', 'Echography', 'Inside Body']
classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
           'Vessel Loop', 'Cadiere Forceps', 'Gauze', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
           'Laparoscopic Fenestrated Forceps', 'Bulldog wire', 'Endobag', 'Veriset', 'Hemolock Clip Applier',
           'Laparoscopic Needle Driver']
classes = ['Tissue', 'Monopolar Curved Scissors', 'Force Bipolar', 'Large Needle Driver', 'Suction',
           'Suture wire', 'Hemolock Clip', 'Fenestrated Bipolar Forceps', 'Suture needle', 'Prograsp Forceps',
           'Vessel Loop', 'Cadiere Forceps', 'Gauze', 'Bulldog clamp', 'Da Vinci trocar', 'Echography',
           'Laparoscopic Fenestrated Forceps', 'Bulldog wire', 'Endobag', 'Veriset', 'Hemolock Clip Applier',
           'Laparoscopic Needle Driver', 'Other instruments']
dataset = RAPN_Dataset(r"/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/train", classes)
image, mask = dataset.__getitem__(10)
print(image.shape, mask.shape)
for i in range(len(classes)):
    print(i, np.count_nonzero(mask[:, :, i] == 1))'''


# Code to understand the conversion of the classes
'''mask = np.array([[0,1,2], [1,4,5], [2,3,6]])
class_values = [0, 1, 3, 4]

masks = [(mask == v) for v in class_values]
print(masks[0].shape)
mask = np.stack(masks, axis=-1).astype('float')
sum = np.sum(mask, axis=2)
print('sum', sum)
result = np.logical_not(np.logical_xor(sum, mask[:, :, 0])).astype('float')
print(result)
print('mask before',mask)
mask[:, :, 0] = result
print('mask after', mask)'''
