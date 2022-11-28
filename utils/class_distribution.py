import cv2
import glob
from alive_progress import alive_bar
import json
import numpy as np
import pandas as pd


def main():
    # Specify the dataset folder
    source_folder = r"/home/kmarc/workspace/nas_private/RAPN100"
    #source_folder = "/Volumes/ORSI/Kevin/RAPN_20procedures"

    # create a color mapping reading the JSON file
    json_path = '../classes.json'
    with open(json_path, 'r') as openfile:
        # Reading from json file
        json_classes = json.load(openfile)

    num_classes = len(json_classes)
    classes = []
    color_mapping = {}
    for i in range(num_classes):
        classes.append(json_classes[i]["name"])

    for cl in classes:
        color_mapping[cl] = json_classes[classes.index(cl)]["color"]

    with open("../config_files/class_mapping.json", 'r') as openfile:
        # Reading from json file
        class_mapping = json.load(openfile)
    class_mapping = class_mapping["instruments"]

    masks = glob.glob(source_folder + '/masks/*/*.png')
    masks.sort()

    count_classes = [0 for i in range(1, 40)]

    with alive_bar(len(masks)) as bar:
        for mask_path in masks:
            # open the mask and retrieve the size
            print(mask_path)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            width, height = mask.shape[:-1]
            flat = mask.reshape(width*height, 3)
            # find the different colors which occur in the mask, to understand the classes present in the image
            colors = np.unique(flat, axis=0)
            # list to check the presence or not of the classes in each image
            class_present = [False for i in range(1, 40)]

            # loop over the image to check the classes present in it
            for col in colors:
                key = str(tuple(col))
                key_reverse = str(tuple(np.flip(col)))
                if key in class_mapping.keys():
                    class_present[class_mapping[key]-1] = True
                elif key_reverse in class_mapping.keys():
                    class_present[class_mapping[key_reverse]-1] = True

            # increment the counter of each class present in the image
            for i in range(len(class_present)):
                if class_present[i]:
                    count_classes[i] += 1

            # Update the progress bar
            bar()

    # ordered list of all the classes
    classes_list =['Bulldog clamp', 'Bulldog wire', 'Cadiere Forceps', 'Catheter',
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

    df.to_excel(source_folder + '/class_distribution.xlsx')


if __name__ == '__main__':
    main()
