import cv2
import glob
from alive_progress import alive_bar
import numpy as np
import pandas as pd
import os
from scipy.stats import ks_2samp, kstest
from scipy.spatial import distance
import math


def main():
    # Specify the dataset folder
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_final_2/masks"
    #source_folder = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/train/masks"
    masks = glob.glob(source_folder + '/*/*.png')
    dir = os.listdir(source_folder)
    dir.sort()

    dir_list = []

    for elem in dir_list:
        print(elem)

    masks.sort()

    # class distribution among the RAPN100 Dataset images, previously computed (to be changed)
    #class_distribution = [30261, 1360, 762, 1874, 11, 61, 784, 117, 217, 5665, 18, 15239, 1500, 536, 7717, 352, 99, 936,
    #                      470, 30, 12650, 1, 0, 60, 22938, 4464, 0, 6, 12130, 37, 5109, 8035, 712, 3035, 0, 1031, 1093,
    #                      22, 244, 31]
    class_distribution = [31871, 1525, 848, 2618, 10, 109, 811, 82, 265, 7496, 821, 14956, 2233, 574, 8069, 35, 80, 980,
                          500, 31, 12776, 4, 0, 61, 24007, 4210, 349, 17, 12328, 40, 5290, 8287, 1087, 3523, 0, 321]
    tot_images = 31871 # total number of images in the dataset (to be updated)
    class_distribution = [c/tot_images for c in class_distribution]
    num_classes = 36

    print(class_distribution)

    #print(ks_2samp(class_distribution,c_dist))

    count_list = []
    for el in [i[4:] for i in dir]:
        st = "/RAPN{el}/".format(el=el)
        count_list.append(len([i for i in masks if st in i]))
    print(dir, count_list)

    distances = []
    for d, c in zip(dir, count_list):
        count_classes = [0 for i in range(0, num_classes)]
        folder_mask = glob.glob(os.path.join(source_folder, d) + '/*.png')
        if len(folder_mask) == 0:
            distances.append(0)
            continue
        print(d)
        for mask_path in folder_mask:
            mask = cv2.imread(mask_path, 0)
            width, height = mask.shape
            flat = mask.reshape(width * height)
            # find the different colors which occur in the mask, to understand the classes present in the image
            classes = np.unique(flat)

            # loop over the image to check the classes present in it
            for c in classes:
                count_classes[c] += 1

        count_classes = [i / len(folder_mask) for i in count_classes]
        dist = distance.euclidean(count_classes, class_distribution)
        print(dist)
        distances.append(dist)

        #print(kstest(count_classes, class_distribution))

    # ordered list of all the classes
    classes_list =['Background', 'Bulldog clamp', 'Bulldog wire', 'Cadiere Forceps', 'Catheter',
     'Drain', 'Endobag', 'Endobag specimen retriever', 'Endobag wire', 'Fenestrated Bipolar Forceps', 'Fibrilar',
     'Force Bipolar', 'Gauze', 'Hemolock Clip Applier', 'Hemolock Clip', 'Inside Body', 'Laparoscopic Clip Applier',
     'Laparoscopic Fenestrated Forceps', 'Laparoscopic Needle Driver', 'Laparoscopic Scissors', 'Large Needle Driver',
     'Left PBP Needle Driver', 'Maryland Bipolar Forceps', 'Metal clip', 'Monopolar Curved Scissors',
     'Prograsp Forceps', 'Right PBP Needle Driver', 'Scissors', 'Suction', 'Surgical_Glove_Tip', 'Suture needle',
     'Suture wire', 'Veriset', 'Vessel Loop', 'Vessel Sealer Extend', 'Echography', 'Da Vinci trocar',
     'Assistant trocar', 'Airseal trocar', 'Foam extruder']

    classes_list = ['Background', 'Bulldog clamp', 'Bulldog wire', 'Cadiere Forceps', 'Da Vinci Obturator',
                    'Drain', 'Endobag', 'Endobag specimen retriever', 'Endobag wire', 'Fenestrated Bipolar Forceps',
                    'Endoscope Trocar',
                    'Force Bipolar', 'Hemostasis', 'Hemolock Clip Applier', 'Hemolock Clip', 'Floseal_Extruder',
                    'Laparoscopic Clip Applier',
                    'Laparoscopic Fenestrated Forceps', 'Laparoscopic Needle Driver', 'Laparoscopic Scissors',
                    'Large Needle Driver',
                    'Tachoseal_introducer', 'Maryland Bipolar Forceps', 'Metal clip', 'Monopolar Curved Scissors',
                    'Prograsp Forceps', 'Airseal trocar', 'Assistant trocar', 'Suction', 'Surgical_Glove_Tip',
                    'Suture needle',
                    'Suture wire', 'Echography', 'Vessel Loop', 'Vessel Sealer Extend', 'Da Vinci trocar']

    # create a Dataframe with the occurencies of each class
    df = pd.DataFrame(distances, index=dir, columns=['Occurencies'])
    print(df)

    df.to_excel("/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_final_2/" + 'directory_distance.xlsx')


if __name__ == '__main__':
    main()
