import os.path

import cv2
import glob
from alive_progress import alive_bar
import json
from utils.semantic_masks import color2class

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

def crop(img, mask, removing_color):

    # compute the contours and the areas of them
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
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
        if count < 0.5*height*width:
            break

    #cv2.imshow('image', img_crop)
    #cv2.waitKey(0)

    return img_crop, mask_crop


def main():

    # Specify the input folder and the output folder
    source_folder = r"/home/kmarc/workspace/nas_private/RAPN100"
    dest_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN"
    create_dataset_tree(source_folder, dest_folder)

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
                       'CT image', '3D-image']
    removing_color = [color_mapping[k] for k in removing_object]

    with open("../config_files/class_mapping.json", 'r') as openfile:
        # Reading from json file
        class_mapping = json.load(openfile)
    class_mapping = class_mapping["instruments"]

    images = glob.glob(source_folder + '/raw_images/*/*.png')
    masks = glob.glob(source_folder + '/masks/*/*.png')
    images.sort()
    masks.sort()

    with alive_bar(len(images)) as bar:
        for im_path, mask_path in zip(images, masks):
            # open the image and retrieve the size
            #im = cv2.cvtColor(cv2.imread(source_folder + "/raw_images/RAPN10/RAPN10_0124.png"), cv2.COLOR_BGR2RGB)
            im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            # mask = cv2.cvtColor(cv2.imread(source_folder + "/annotations/RAPN10/RAPN10_0124.png___fuse.png"), cv2.COLOR_BGR2RGB)
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
                im_path = dest_folder + '/raw_images/' + procedure + '/' + os.path.basename(im_path)
                cv2.imwrite(im_path, cv2.cvtColor(im_cropped, cv2.COLOR_BGR2RGB))
                # Generate the corresponding class mask (= matrix with value related to class for each pixel),
                # according to the "class_mapping.json" file
                mask_cropped = color2class(mask_cropped, class_mapping)
                mask_path = im_path.replace('raw_images', 'masks')
                cv2.imwrite(mask_path, cv2.cvtColor(mask_cropped, cv2.COLOR_BGR2RGB))
                
            # Update the progress bar
            bar()


if __name__ == '__main__':

    main()
