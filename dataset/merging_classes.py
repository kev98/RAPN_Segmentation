"""
Python file to prepare the final RAPN100 dataset, with the following permanent merging:
- Inside body —> Background
- Gauzes, Hemostatic agens, veriset, fibrillar —> hemostasis
"""

import os.path
import cv2
import glob
from alive_progress import alive_bar
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


def main():
    # Specify the input folder and the output folder "directory of the dataset/(train or val or test)"
    folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_bis/test"
    #folder = r'/Volumes/ORSI/Kevin/RAPN100/train'

    masks = glob.glob(folder + '/masks/*/*.png')
    masks.sort()

    with alive_bar(len(masks)) as bar:
        for mask_path in masks:
            # open the image and the mask and retrieve the size
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            mask[mask == 15] = 0 # set inside body to background
            mask[mask == 41] = 12 # set Hemostatic angens to Hemostasis
            mask[mask == 32] = 12 # set Veriset to Hemostasis
            mask[mask == 10] = 12 # set fibrillar to Hemostasis

            cv2.imwrite(mask_path, cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

            # Update the progress bar
            bar()


if __name__ == '__main__':
    main()
