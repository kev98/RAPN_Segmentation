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
    source_folder = r"/home/kmarc/workspace/nas_private/RAPN100"
    dest_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN"
    #source_folder = r"/Volumes/ORSI/Kevin/RAPN100_prova"
    #dest_folder = r'/Volumes/ORSI/Kevin/RAPN100'
    # create_dataset_tree(source_folder, dest_folder)

    list_images = ["RAPN102_0278.png", "RAPN102_0279.png", "RAPN103_0179.png", "RAPN103_0180.png", "RAPN103_0188.png",
     "RAPN103_0253.png", "RAPN103_0254.png", "RAPN103_0255.png", "RAPN113_0321.png", "RAPN113_0322.png",
     "RAPN113_0323.png", "RAPN113_0324.png", "RAPN113_0325.png", "RAPN113_0326.png", "RAPN113_0327.png",
     "RAPN21_0379.png", "RAPN21_0380.png", "RAPN21_0381.png", "RAPN21_0382.png", "RAPN21_0383.png", "RAPN21_0384.png",
     "RAPN21_0385.png", "RAPN21_0386.png", "RAPN21_0387.png", "RAPN21_0388.png", "RAPN21_0389.png", "RAPN21_0390.png",
     "RAPN21_0391.png", "RAPN21_0392.png", "RAPN30_0362.png", "RAPN30_0363.png", "RAPN30_0374.png", "RAPN30_0375.png",
     "RAPN30_0376.png", "RAPN30_0377.png", "RAPN32_0359.png", "RAPN32_0360.png", "RAPN32_0361.png", "RAPN32_0362.png",
     "RAPN32_0363.png", "RAPN32_0364.png", "RAPN32_0365.png", "RAPN32_0366.png", "RAPN32_0368.png", "RAPN32_0369.png",
     "RAPN32_0370.png", "RAPN32_0371.png", "RAPN32_0372.png", "RAPN52_0239.png", "RAPN52_0240.png", "RAPN79_0213.png",
     "RAPN79_0214.png", "RAPN79_0215.png", "RAPN79_0216.png", "RAPN79_0217.png", "RAPN79_0218.png", "RAPN79_0219.png",
     "RAPN79_0220.png", "RAPN79_0221.png", "RAPN79_0222.png", "RAPN79_0223.png", "RAPN79_0224.png", "RAPN79_0225.png",
     "RAPN79_0226.png", "RAPN79_0227.png", "RAPN79_0228.png", "RAPN79_0229.png", "RAPN79_0231.png", "RAPN89_0307.png",
     "RAPN89_0308.png", "RAPN89_0309.png", "RAPN89_0310.png", "RAPN89_0311.png", "RAPN89_0321.png", "RAPN89_0322.png",
     "RAPN89_0323.png", "RAPN89_0324.png", "RAPN90_0122.png", "RAPN97_0207.png", "RAPN98_0172.png", "RAPN98_0173.png",
     "RAPN98_0180.png", "RAPN98_0181.png", "RAPN98_0182.png", "RAPN98_0183.png", "RAPN98_0184.png",
     "RAPN110_0117.png", "RAPN110_0153.png", "RAPN110_0161.png", "RAPN110_0203.png", "RAPN110_0205.png",
     "RAPN110_0206.png", "RAPN110_0211.png", "RAPN110_0213.png", "RAPN110_0214.png", "RAPN110_0215.png",
     "RAPN110_0216.png", "RAPN110_0217.png", "RAPN110_0218.png", "RAPN110_0219.png", "RAPN110_0220.png",
     "RAPN110_0221.png", "RAPN110_0222.png", "RAPN110_0223.png", "RAPN110_0224.png", "RAPN110_0225.png",
     "RAPN110_0226.png", "RAPN110_0227.png", "RAPN110_0228.png", "RAPN110_0229.png", "RAPN110_0230.png",
     "RAPN110_0231.png", "RAPN110_0232.png", "RAPN19_0474.png", "RAPN59_0095.png", "RAPN59_0104.png", "RAPN59_0106.png",
     "RAPN59_0107.png", "RAPN59_0108.png", "RAPN59_0110.png", "RAPN59_0111.png", "RAPN59_0112.png", "RAPN59_0113.png",
     "RAPN59_0115.png", "RAPN59_0116.png", "RAPN59_0117.png", "RAPN66_0105.png", "RAPN66_0106.png", "RAPN66_0107.png",
     "RAPN66_0108.png", "RAPN66_0109.png", "RAPN66_0110.png", "RAPN66_0111.png", "RAPN66_0112.png", "RAPN66_0113.png",
     "RAPN66_0114.png", "RAPN66_0115.png", "RAPN66_0116.png", "RAPN66_0118.png", "RAPN66_0119.png", "RAPN66_0120.png",
     "RAPN66_0121.png", "RAPN66_0122.png", "RAPN66_0123.png", "RAPN66_0124.png", "RAPN66_0125.png", "RAPN66_0129.png",
     "RAPN66_0130.png", "RAPN66_0133.png", "RAPN66_0134.png", "RAPN66_0135.png", "RAPN66_0136.png", "RAPN66_0138.png",
     "RAPN67_0123.png", "RAPN67_0193.png", "RAPN67_0194.png", "RAPN67_0195.png", "RAPN67_0200.png", "RAPN67_0201.png",
     "RAPN67_0204.png", "RAPN67_0205.png", "RAPN96_0073.png"]


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
    print(class_mapping)

    '''images = glob.glob(source_folder + '/raw_images/*/*.png')
    masks = glob.glob(source_folder + '/masks/*/*.png')
    images.sort()
    masks.sort()'''
    val_set = ['RAPN102', 'RAPN108', 'RAPN19', 'RAPN41', 'RAPN45', 'RAPN7', 'RAPN76', 'RAPN79', 'RAPN8', 'RAPN81',
               'RAPN89', 'RAPN92', 'RAPN95', 'RAPN98']
    test_set = ['RAPN115', 'RAPN20', 'RAPN36', 'RAPN47', 'RAPN48', 'RAPN50', 'RAPN91', 'RAPN96']

    with alive_bar(len(list_images)) as bar:
        for i in list_images:
            procedure = i.split('_')[0]
            im_path = os.path.join(source_folder, 'raw_images', procedure, i)
            print(im_path)
            mask_path = im_path.replace('raw_images', 'masks')
            mask_path = mask_path + '___fuse.png'
            print(mask_path)
            # open the image and the mask and retrieve the size
            if os.path.isfile(im_path):
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
                    # print(procedure)
                    if procedure in val_set:
                        dest = os.path.join(dest_folder, 'val')
                    elif procedure in test_set:
                        dest = os.path.join(dest_folder, 'test')
                    else:
                        dest = os.path.join(dest_folder, 'train')
                    im_path = dest + '/raw_images/' + procedure + '/' + i
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