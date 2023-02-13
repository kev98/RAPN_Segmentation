"""
Script to remove from the images of the RAPN100 the HUD if it's present)
"""

import os.path
import cv2
import glob
from alive_progress import alive_bar
import json
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils.semantic_masks import color2class


# Function which crop the part of the image we are interested in
def crop(img, mask):

    mask_crop = mask[:-64, :]
    img_crop = img[:-64, :]

    return img_crop, mask_crop


def main():
    # Specify the input folder
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_2"
    source_folder = r"/Volumes/ORSI/Kevin/Segmentation_Dataset_RAPN_2/train"

    images = glob.glob(source_folder + '/raw_images/*/*.png')
    masks = glob.glob(source_folder + '/masks/*/*.png')
    images.sort()
    masks.sort()

    with alive_bar(len(images)) as bar:
        for im_path, mask_path in zip(images, masks):
            # open the image and the mask and retrieve the size
            im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            width, height = mask.shape[:-1]

            # check if the image has to be cropped
            if width > 900 and height > 900:
                # crop the image and the mask and save them
                im_cropped, mask_cropped = crop(im, mask)
                print(im.shape, im_cropped.shape)
                print(mask.shape, mask_cropped.shape)
                cv2.imwrite(im_path, cv2.cvtColor(im_cropped, cv2.COLOR_BGR2RGB))
                cv2.imwrite(mask_path, cv2.cvtColor(mask_cropped, cv2.COLOR_BGR2RGB))
            # Update the progress bar
            bar()


if __name__ == '__main__':
    main()
