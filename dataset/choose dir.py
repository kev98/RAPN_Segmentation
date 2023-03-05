import cv2
import glob
from alive_progress import alive_bar
import numpy as np
import pandas as pd
import os
from scipy.stats import ks_2samp, kstest
from scipy.spatial import distance
import math
from itertools import combinations

def main():
    # Specify the dataset folder
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/masks"
    #source_folder = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/train/masks"
    masks = glob.glob(source_folder + '/*/*.png')
    #dir = ['RAPN38', 'RAPN7', 'RAPN104', 'RAPN12', 'RAPN115', 'RAPN39', 'RAPN34']
    dir = ['RAPN91', 'RAPN20', 'RAPN96', 'RAPN102', 'RAPN47', 'RAPN41', 'RAPN87', 'RAPN92', 'RAPN81', 'RAPN95',
            'RAPN28', 'RAPN19']
    dir.sort()

    masks.sort()

    # class distribution among the RAPN100 Dataset images, previously computed
    class_distribution = [30261, 1360, 762, 1874, 11, 61, 784, 117, 217, 5665, 18, 15239, 1500, 536, 7717, 352, 99, 936,
                          470, 30, 12650, 1, 0, 60, 22938, 4464, 0, 6, 12130, 37, 5109, 8035, 712, 3035, 0, 1031, 1093,
                          22, 244, 31]
    class_distribution = [c/30438 for c in class_distribution]

    print(class_distribution)

    #print(ks_2samp(class_distribution,c_dist))

    count_list = []
    for el in [i[4:] for i in dir]:
        st = "/RAPN{el}/".format(el=el)
        count_list.append(len([i for i in masks if st in i]))
    print(dir, count_list)

    # ordered list of all the classes
    classes_list =['Background', 'Bulldog clamp', 'Bulldog wire', 'Cadiere Forceps', 'Catheter',
     'Drain', 'Endobag', 'Endobag specimen retriever', 'Endobag wire', 'Fenestrated Bipolar Forceps', 'Fibrilar',
     'Force Bipolar', 'Gauze', 'Hemolock Clip Applier', 'Hemolock Clip', 'Inside Body', 'Laparoscopic Clip Applier',
     'Laparoscopic Fenestrated Forceps', 'Laparoscopic Needle Driver', 'Laparoscopic Scissors', 'Large Needle Driver',
     'Left PBP Needle Driver', 'Maryland Bipolar Forceps', 'Metal clip', 'Monopolar Curved Scissors',
     'Prograsp Forceps', 'Right PBP Needle Driver', 'Scissors', 'Suction', 'Surgical_Glove_Tip', 'Suture needle',
     'Suture wire', 'Veriset', 'Vessel Loop', 'Vessel Sealer Extend', 'Echography', 'Da Vinci trocar',
     'Assistant trocar', 'Airseal trocar', 'Foam extruder']

    distances = []
    count_classes = [0 for i in range(0, 40)]
    count = 0
    for d, c in zip(dir, count_list):
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
            count += 1

        #print(kstest(count_classes, class_distribution))
    print(count)

    for i in range(len(classes_list)):
        print(classes_list[i], count_classes[i]/count)
