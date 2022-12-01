import cv2
import glob
from alive_progress import alive_bar
import json
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance


def main():
    # Specify the dataset folder
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/masks"
    #source_folder = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/train/masks"

    dir_list = ['RAPN91', 'RAPN20', 'RAPN96', 'RAPN47', 'RAPN50', 'RAPN48', 'RAPN115', 'RAPN36']
    class_distribution = [30261, 1360, 762, 1874, 11, 61, 784, 117, 217, 5665, 18, 15239, 1500, 536, 7717, 352, 99, 936,
                          470, 30, 12650, 1, 0, 60, 22938, 4464, 0, 6, 12130, 37, 5109, 8035, 712, 3035, 0, 1031, 1093,
                          22, 244, 31]

    #masks = glob.glob(source_folder + '/masks/*/*.png')
    #masks.sort()

    count_classes = [0 for i in range(0, 40)]

    with alive_bar(len(dir_list)) as bar:
        for dir in dir_list:
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

        # Update the progress bar
        bar()

    dist = distance.euclidean([i/3197 for i in count_classes], [i/30438 for i in class_distribution])

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
    df = pd.DataFrame(count_classes, index=classes_list, columns=['Occurencies'])
    print(df)
    print('The euclidean distance between the whole dataset and the test set is: ', dist)

    df.to_excel(source_folder + '/class_distribution_testset.xlsx')


if __name__ == '__main__':
    main()