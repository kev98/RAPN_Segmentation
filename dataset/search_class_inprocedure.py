import os
import glob
import cv2
import argparse

# Parse input argument
parser = argparse.ArgumentParser(description="Script to check shape of the images and search for an image with a specific class inside",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--procedure", help="number of the procedure")
parser.add_argument("-c", "--class", help="number of the class")
parser.add_argument("-s", "--set", help="in which set is the class")
args = parser.parse_args()
config = vars(args)

# Set the variables
procedure = config['procedure']
cl = int(config['class'])
s = config['set']

#procedure_path = '/Volumes/ORSI/Ila_RAPN/masks/RAPN41'
procedure_path = os.path.join('/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN', s, 'masks/RAPN') + procedure

images_path = glob.glob(procedure_path + '/*')

img_list = []

for im in images_path:
     image = cv2.imread(im)
     if image.shape[0] < 500 or image.shape[1] < 500:
        print(im, image.shape)
     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     if cl in image:
          img_list.append(im.split('/')[-1])

print('The class ' + str(cl) + ' in the procedure number ' + procedure + ' is present in', len(img_list), 'images')
print(img_list)

