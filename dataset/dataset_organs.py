import glob
from torch.utils.data import Dataset as BaseDataset
import cv2
import os
# to avoid multiprocessing deadlocks
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np


class DSAD_Dataset(BaseDataset):
    """Dresden Surgucal Anatomy Dataset. Read images and corresponding masks, apply augmentation and preprocessing transformations.

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

        self.classes = classes
        self.images = glob.glob(os.path.join(images_dir, '*/image*'))
        self.masks = []
        msks = glob.glob(os.path.join(images_dir, '*/mask*'))
        for cl in classes[:-1]:
            l = [m for m in msks if str(cl) in m]
            l.sort()
            self.masks.append(l)
        self.images.sort()
        print('number of images: ',len(self.images))
        print('number of masks: ', len(self.masks[0]))

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read image and mask
        #print('image: ', self.images[i])
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(self.images[i])

        masks = [cv2.imread(self.masks[c][i], 0)==255 for c in range(len(self.classes)-1)]
        mask = np.stack(masks, axis=-1).astype('float')
        #print('image: ', image.shape)
        #print('mask: ', mask.shape)

        sum = np.ones_like(mask[:,:,0]) - np.sum(mask, axis=2)
        sum [sum < 0] = 0 # because I found that there are some images in which a pixel belongs to two classes
        #print(sum.shape)
        mask = np.concatenate((mask, sum[..., None]), axis=-1).astype('float')


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
'''classes = ['abdominal_wall', 'colon', 'liver', 'pancreas', 'small_intestine', 'spleen', 'stomach', 'other']

dataset = DSAD_Dataset(r"/Volumes/TOSHIBA EXT/DSAD_Dataset/multilabel", classes)
image, mask = dataset.__getitem__(80)
print(image.shape, mask.shape)
for i in range(len(classes)):
    print(i, np.count_nonzero(mask[:, :, i] == 1))
print('-1', np.count_nonzero(mask[:, :, 7] == -1))'''
