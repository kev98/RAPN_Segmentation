"""
Script to remove from the images of the RAPN100 the HUD if it's present)
"""

import os.path
import cv2
import glob
from alive_progress import alive_bar
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


# Function which crop the part of the image we are interested in
def crop(img, mask):

    mask_crop = mask[:-64, :]
    img_crop = img[:-64, :]

    return img_crop, mask_crop


def main():
    # Specify the input folder
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_2/train"
    #source_folder = r"/Volumes/ORSI/Kevin/Segmentation_Dataset_RAPN_2/train"

    # List of procedures in which is present the HUD
    procedures = ['RAPN7', 'RAPN8', 'RAPN10', 'RAPN11', 'RAPN12', 'RAPN33', 'RAPN34', 'RAPN35',
                  'RAPN36', 'RAPN37', 'RAPN38', 'RAPN39', 'RAPN41', 'RAPN54', 'RAPN61', 'RAPN62',
                  'RAPN76', 'RAPN77', 'RAPN78', 'RAPN79', 'RAPN81', 'RAPN82', 'RAPN85', 'RAPN86',
                  'RAPN87', 'RAPN88', 'RAPN89', 'RAPN90', 'RAPN92', 'RAPN93', 'RAPN94', 'RAPN95',
                  'RAPN96', 'RAPN97', 'RAPN98', 'RAPN99', 'RAPN100', 'RAPN102', 'RAPN103', 'RAPN104',
                  'RAPN107', 'RAPN108', 'RAPN109', 'RAPN110', 'RAPN111', 'RAPN114', 'RAPN115', 'RAPN116',
                  'RAPN117', 'RAPN118', 'RAPN119', 'RAPN120', 'RAPN121', 'RAPN122']

    for p in procedures:
        images = glob.glob(source_folder + '/raw_images/' + p + '/*.png')
        masks = glob.glob(source_folder + '/masks/' + p + '/*.png')
        images.sort()
        masks.sort()

        if len(images) > 0:
            print(p)
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
                        #print(im.shape, im_cropped.shape)
                        #print(mask.shape, mask_cropped.shape)
                        cv2.imwrite(im_path, cv2.cvtColor(im_cropped, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(mask_path, cv2.cvtColor(mask_cropped, cv2.COLOR_BGR2RGB))
                    # Update the progress bar
                    bar()


if __name__ == '__main__':
    main()
