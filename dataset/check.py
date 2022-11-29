import cv2
import glob
from alive_progress import alive_bar
import numpy as np
import pandas as pd
import os
from scipy.stats import ks_2samp, kstest
from scipy.spatial import distance
import math

def mean( hist ):
    mean = 0.0
    for i in hist:
        mean += i
    mean/= len(hist)
    return mean

def bhatta ( hist1,  hist2):
    # calculate mean of hist1
    h1_ = mean(hist1)

    # calculate mean of hist2
    h2_ = mean(hist2)

    # calculate score
    score = 0
    for i in range(hist1.shape[0]):
        score += math.sqrt( hist1[i] * hist2[i] )
    # print h1_,h2_,score;
    score = math.sqrt( 1 - ( 1 / math.sqrt(h1_*h2_*8*8) ) * score )
    return score

def main():
    # Specify the dataset folder
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/masks"
    #source_folder = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/train/masks"
    masks = glob.glob(source_folder + '/*/*.png')
    dir = ['RAPN38', 'RAPN7', 'RAPN104', 'RAPN12', 'RAPN115', 'RAPN39', 'RAPN34']
    #dir = ['RAPN7', 'RAPN10']
    dir.sort()

    masks.sort()

    # class distribution among the RAPN100 Dataset images, previously computed
    class_distribution = [30261, 22938, 15239, 12650, 12130, 8035, 7717, 5665, 5109, 4464, 3035, 1874, 1500, 1360, 1093,
                          1031, 936 ,784, 762, 712, 536, 470, 352, 244, 217, 117, 99, 61, 60, 37, 31, 30, 22, 18, 11, 6,
                          1, 0, 0, 0]
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

        #print(kstest(count_classes, class_distribution))

    for i in range(len(classes_list)):
        print(classes_list[i], count_classes[i])


if __name__ == '__main__':
    main()
