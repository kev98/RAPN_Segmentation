import cv2
import glob
from alive_progress import alive_bar
import json
import numpy as np
import pandas as pd
import os


def main():
    # Specify the dataset folder
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/masks"
    #source_folder = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/train/masks"

    dir_list = os.listdir(source_folder)

    #masks = glob.glob(source_folder + '/masks/*/*.png')
    #masks.sort()

    tot_count = []

    with alive_bar(len(dir_list)) as bar:
        for dir in dir_list:
            count_classes = [0 for i in range(0, 40)]
            masks = glob.glob(os.path.join(source_folder, dir) + '/*.png')
            masks.sort()
            for mask_path in masks:
                # open the mask and retrieve the size
                print(mask_path)
                mask = cv2.imread(mask_path, 0)
                width, height = mask.shape
                flat = mask.reshape(width*height)
                # find the different colors which occur in the mask, to understand the classes present in the image
                classes = np.unique(flat)

                # loop over the image to check the classes present in it
                for c in classes:
                    count_classes[c] += 1
            tot_count.append(count_classes)

        # Update the progress bar
        bar()
    tot_count = np.array(tot_count)
    tot_count = tot_count.transpose(1,0)
    print(tot_count)

    # ordered list of all the classes
    classes_list =['Background', 'Bulldog clamp', 'Bulldog wire', 'Cadiere Forceps', 'Catheter',
     'Drain', 'Endobag', 'Endobag specimen retriever', 'Endobag wire', 'Fenestrated Bipolar Forceps', 'Fibrilar',
     'Force Bipolar', 'Gauze', 'Hemolock Clip Applier', 'Hemolock Clip', 'Inside Body', 'Laparoscopic Clip Applier',
     'Laparoscopic Fenestrated Forceps', 'Laparoscopic Needle Driver', 'Laparoscopic Scissors', 'Large Needle Driver',
     'Left PBP Needle Driver', 'Maryland Bipolar Forceps', 'Metal clip', 'Monopolar Curved Scissors',
     'Prograsp Forceps', 'Right PBP Needle Driver', 'Scissors', 'Suction', 'Surgical_Glove_Tip', 'Suture needle',
     'Suture wire', 'Veriset', 'Vessel Loop', 'Vessel Sealer Extend', 'Echography', 'Da Vinci trocar',
     'Assistant trocar', 'Airseal trocar', 'Foam extruder']

    # create a Dataframe with the occurencies of each class
    df = pd.DataFrame(tot_count, index=classes_list, columns=dir_list)
    print(df)

    df.to_excel(source_folder + '/class_distribution.xlsx')


if __name__ == '__main__':
    main()