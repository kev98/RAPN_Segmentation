import glob
import cv2

procedure_path = '/Volumes/ORSI/Ila_RAPN/masks/RAPN41'

cl = 6 # class "Endobag"

images_path = glob.glob(procedure_path + '/*')

img_list = []

for im in images_path:
     image = cv2.imread(im)
     print(im, image.shape)
     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     if cl in image:
          img_list.append(im)

print(img_list)

