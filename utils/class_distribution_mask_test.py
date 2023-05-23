# Script to compute the classes' occurences and distribution of a specific set of the final dataset

import cv2
import glob
from alive_progress import alive_bar
import numpy as np
import pandas as pd
from scipy.spatial import distance


def main():
    # Specify the dataset folder
    #source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_tris/test/masks"
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_final_2/train/masks"
    #source_folder = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/train/masks"

    #dir_list = ['RAPN91', 'RAPN20', 'RAPN96', 'RAPN47', 'RAPN50', 'RAPN48', 'RAPN115', 'RAPN36']
    #dir_list = ['RAPN102', 'RAPN41', 'RAPN92', 'RAPN81', 'RAPN95', 'RAPN19', 'RAPN89', 'RAPN45', 'RAPN108', 'RAPN79',
                #'RAPN98', 'RAPN76', 'RAPN7', 'RAPN8']
    #class_distribution = [30261, 1360, 762, 1874, 11, 61, 784, 117, 217, 5665, 18, 15239, 1500, 536, 7717, 352, 99, 936,
    #                      470, 30, 12650, 1, 0, 60, 22938, 4464, 0, 6, 12130, 37, 5109, 8035, 712, 3035, 0, 1031, 1093,
    #                      22, 244, 31]
    # class occurences in the entire final dataset
    class_distribution = [31811, 1462, 823, 2402, 0, 78, 811, 112, 236, 7099, 0, 15277, 2160, 580, 8058, 0, 86, 974,
                          503, 29, 12757, 0, 0, 62, 24062, 4205, 0, 0, 12437, 39, 5203, 8166, 0, 3411, 0, 1060, 1136,
                          15, 291, 33, 76, 0]
    class_distribution = [31871, 1525, 848, 2618, 10, 109, 811, 82, 265, 7496, 821, 14956, 2233, 574, 8069, 35, 80, 980,
                          500, 31, 12776, 4, 0, 61, 24007, 4210, 349, 17, 12328, 40, 5290, 8287, 1087, 3523, 0, 321]
    #masks = glob.glob(source_folder + '/masks/*/*.png')
    #masks.sort()

    num_classes = 36  # number of classes (can change)
    count_classes = [0 for i in range(0, num_classes)]
    masks = glob.glob(source_folder + '/*/*.png')
    masks.sort()

    with alive_bar(len(masks)) as bar:
        #for dir in dir_list:
        for mask_path in masks:
            # open the mask and retrieve the size
            #print(mask_path)
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

    set_images = len(masks) # images in the set we are evaluating
    dataset_images = 31871 # total number of images in the entire final dataset
    dist = distance.euclidean([i/set_images for i in count_classes], [i/dataset_images for i in class_distribution])

    # ordered list of all the classes
    '''classes_list =['Background', 'Bulldog clamp', 'Bulldog wire', 'Cadiere Forceps', 'Catheter',
     'Drain', 'Endobag', 'Endobag specimen retriever', 'Endobag wire', 'Fenestrated Bipolar Forceps', 'Fibrilar',
     'Force Bipolar', 'Gauze', 'Hemolock Clip Applier', 'Hemolock Clip', 'Inside Body', 'Laparoscopic Clip Applier',
     'Laparoscopic Fenestrated Forceps', 'Laparoscopic Needle Driver', 'Laparoscopic Scissors', 'Large Needle Driver',
     'Left PBP Needle Driver', 'Maryland Bipolar Forceps', 'Metal clip', 'Monopolar Curved Scissors',
     'Prograsp Forceps', 'Right PBP Needle Driver', 'Scissors', 'Suction', 'Surgical_Glove_Tip', 'Suture needle',
     'Suture wire', 'Veriset', 'Vessel Loop', 'Vessel Sealer Extend', 'Echography', 'Da Vinci trocar',
     'Assistant trocar', 'Airseal trocar', 'Foam extruder']'''

    '''classes_list =['Background', 'Bulldog clamp', 'Bulldog wire', 'Cadiere Forceps', 'Catheter',
     'Drain', 'Endobag', 'Endobag specimen retriever', 'Endobag wire', 'Fenestrated Bipolar Forceps', 'null',
     'Force Bipolar', 'Hemostasis', 'Hemolock Clip Applier', 'Hemolock Clip', 'null', 'Laparoscopic Clip Applier',
     'Laparoscopic Fenestrated Forceps', 'Laparoscopic Needle Driver', 'Laparoscopic Scissors', 'Large Needle Driver',
     'Left PBP Needle Driver', 'Maryland Bipolar Forceps', 'Metal clip', 'Monopolar Curved Scissors',
     'Prograsp Forceps', 'Right PBP Needle Driver', 'Scissors', 'Suction', 'Surgical_Glove_Tip', 'Suture needle',
     'Suture wire', 'null', 'Vessel Loop', 'Vessel Sealer Extend', 'Echography', 'Da Vinci trocar',
     'Assistant trocar', 'Airseal trocar', 'Foam extruder', 'Foam', 'null']
    '''
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
    df = pd.DataFrame(count_classes, index=classes_list, columns=['Occurencies'])
    print(df)
    print('The euclidean distance between the whole dataset and the test set is: ', dist)

    df.to_excel(source_folder + '/class_distribution_testset_final.xlsx')


if __name__ == '__main__':
    main()
