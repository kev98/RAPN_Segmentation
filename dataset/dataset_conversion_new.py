"""
Script to prepare the RAPN100 Dataset for the semantic segmentation. The script create a new usable dataset only
with the images useful for the Instrument Segmentation task, cropped in the rigth way
- Step 1. Cleaning of the dataset: removal of all the images which contain only classes not useful for the segmentation
  of the instruments (like 'Outside body', 'Ecography image, 'Color bar' etc...)
- Step 2. Cropping of the images: compute the contours of the images, then find the biggest contour which
  which contains classes of interest and crop it
- Step 3. Conversion of the mask: mask are converted from RGB to one channel images, mapping each pixel to the
  corresponding class; furthermore different instances of the same class are merged into a single class
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
from utils.semantic_masks import color2classopt


# Function to create the directory tree of the dataset, if it's necessary
def create_dataset_tree(source, dest):

    if not os.path.exists(dest):
        os.mkdir(dest)
        os.mkdir(os.path.join(dest, 'raw_images'))
        os.mkdir(os.path.join(dest, 'masks'))

    dir = os.listdir(source + '/raw_images')
    for d in dir:
        img_path = os.path.join((dest + '/raw_images'), d)
        mask_path = os.path.join((dest + '/masks'), d)
        if not os.path.exists(img_path):
            os.mkdir(img_path)
            os.mkdir(mask_path)


def main():

    # Specify the input folder and the output folder
    source_folder = r"/home/kmarc/workspace/nas_private/RAPN100_final"
    dest_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN_final_2"
    create_dataset_tree(source_folder, dest_folder)

    # create a color mapping reading the JSON file
    json_path = '../classes_new.json'
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
    print(color_mapping)

    with open("../config_files/class_mapping_new.json", 'r') as openfile:
        # Reading from json file
        class_mapping = json.load(openfile)
    class_mapping = class_mapping["instruments"]

    images = glob.glob(source_folder + '/raw_images/*/*.png')
    masks = glob.glob(source_folder + '/masks/*/*.png')
    images.sort()
    masks.sort()

    with alive_bar(len(images)) as bar:
        i = 0
        for im_path, mask_path in zip(images, masks):
            # open the image and the mask and retrieve the size
            i = i+1
            print(i)
            im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            procedure = im_path.split('/')[-2]
            im_path = dest_folder + '/raw_images/' + procedure + '/' + os.path.basename(im_path)

            cv2.imwrite(im_path, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            # Generate the corresponding class mask (= matrix with value related to class for each pixel),
            # according to the "class_mapping.json" file
            mask_converted = color2classopt(mask, class_mapping)
            mask_path = im_path.replace('raw_images', 'masks')
            cv2.imwrite(mask_path, cv2.cvtColor(mask_converted, cv2.COLOR_BGR2RGB))
                
            # Update the progress bar
            bar()


if __name__ == '__main__':

    main()
