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
from utils.semantic_masks import color2class


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


# Function which crop the part of the image we are interested in
def crop(img, mask, removing_color):
    # compute the contours and the areas of them
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]

    if len(areas) == 0:
        return img

    # order the contours in descending order based on the value of their area
    index = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)

    # find the part of the image that we need to crop
    for i in index:
        cnt = contours[i]
        # img_c = cv2.drawContours(img, contours, max_index, color=[0,255,0], thickness=2)
        x, y, w, h = cv2.boundingRect(cnt)
        mask_crop = mask[y:y + h, x:x + w]
        img_crop = img[y:y + h, x:x + w]
        width, height = mask_crop.shape[:-1]

        count = 0
        for y in range(height):
            for x in range(width):
                b, g, r = mask_crop[x, y]
                color = f"#{b:02x}{g:02x}{r:02x}"
                if color in removing_color:
                    count += 1

        # if the bounding box considered contains mainly classes we are interested in
        if count < 0.5 * height * width:
            break

    # cv2.imshow('image', img_crop)
    # cv2.waitKey(0)

    return img_crop, mask_crop


def main():
    # Specify the input folder and the output folder
    source_folder = r"/home/kmarc/workspace/nas_private/RAPN100_2"
    dest_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN"
    #create_dataset_tree(source_folder, dest_folder)

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
    print(color_mapping)

    # list of object which aren't interesting for us
    removing_object = ['No_Internet', 'Other_computer_image', 'Outside Body',
                       'Inside Trocar', 'Image thing', 'ICG', 'Echography image', 'Color Bars',
                       'CT image', '3D-image', 'Augmented Reality Image']
    removing_color = [color_mapping[k] for k in removing_object]

    with open("../config_files/class_mapping.json", 'r') as openfile:
        # Reading from json file
        class_mapping = json.load(openfile)
    class_mapping = class_mapping["instruments"]

    images = glob.glob(source_folder + '/raw_images/*/*.png')
    masks = glob.glob(source_folder + '/masks/*/*.png')
    images.sort()
    masks.sort()
    val_set = ['RAPN102', 'RAPN108','RAPN19', 'RAPN41', 'RAPN45', 'RAPN7', 'RAPN76', 'RAPN79', 'RAPN8', 'RAPN81', 'RAPN89', 'RAPN92', 'RAPN95', 'RAPN98']
    test_set = ['RAPN115', 'RAPN20', 'RAPN36', 'RAPN47', 'RAPN48', 'RAPN50', 'RAPN91', 'RAPN96']

    with alive_bar(len(images)) as bar:
        for im_path, mask_path in zip(images, masks):
            # open the image and the mask and retrieve the size
            im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            width, height = mask.shape[:-1]

            # loop over the image to check if has to be deleted or not, based on the classes present in it
            remove_image = True
            removing_color.append('#000000')
            for y in range(height):
                for x in range(width):
                    b, g, r = mask[x, y]
                    color = f"#{b:02x}{g:02x}{r:02x}"
                    if color not in removing_color:
                        remove_image = False
            removing_color.remove('#000000')

            # crop the image and the mask and save them
            if not remove_image:
                print(im_path)
                im_cropped, mask_cropped = crop(im, mask, removing_color)
                procedure = im_path.split('/')[-2]
                #print(procedure)
                if procedure in val_set:
                    dest = os.path.join(dest_folder, 'val')
                elif procedure in test_set:
                    dest = os.path.join(dest_folder, 'test')
                else:
                    dest = os.path.join(dest_folder, 'train')
                im_path = dest + '/raw_images/' + procedure + '/' + os.path.basename(im_path) 
                print(im_path)
                cv2.imwrite(im_path, cv2.cvtColor(im_cropped, cv2.COLOR_BGR2RGB))
                # Generate the corresponding class mask (= matrix with value related to class for each pixel),
                # according to the "class_mapping.json" file
                mask_cropped = color2class(mask_cropped, class_mapping)
                mask_path = im_path.replace('raw_images', 'masks')
                cv2.imwrite(mask_path, cv2.cvtColor(mask_cropped, cv2.COLOR_BGR2RGB))
                print("FATTO")

            # Update the progress bar
            bar()


if __name__ == '__main__':
    main()
