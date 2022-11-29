import os.path

import cv2
import numpy as np
import glob
from alive_progress import alive_bar

# Define the mapping of the classes to their respective color in RGB
'''class_color_mapping = {
    0: (0, 0, 0),           # Background - BLACK
    1: (112, 62, 66),       # Monopolar Curved Scissors - BORDEAUX
    2: (152, 1, 130)        # Suction - MAGENTA
}'''

# binary
'''class_color_mapping = {
    0: (0, 0, 0),           # Background - BLACK
    1: (0, 130, 200),       # Instruments
}'''

# da modificare !!!!!
def class2color(mask, class_color_mapping):
    """
    Function which generates a colored mask based on the input class value mask
    :param mask:Aa mask where each class has its own integer value
    :return: A mask where each class has its own color
    """

    # Initialize a blank image
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])

    # Iterate over the possible class values, as defined in class_color_mapping
    for i in class_color_mapping:
        # Assign the color corresponding to the class value to the appropriate pixels in the blank image
        color_mask[np.where(mask == i)] = class_color_mapping[i]

    return color_mask


def color2class(color_mask, class_color_mapping):
    """
    Function which generates a class value mask based on the input colored mask
    :param color_mask: A mask where each class has its own color
    :return: A mask where each class has its own integer value
    """

    # Initialize a blank image
    width, height = color_mask.shape[:-1]
    mask = np.zeros([width, height], dtype=np.uint8)

    # Iterate over the pixels of the image
    for y in range(height):
        for x in range(width):
            # Assign the class value corresponding to the color to the appropriate pixels in the blank image
            key = str(tuple(color_mask[x, y]))
            key_reverse = str(tuple(np.flip(color_mask[x, y])))
            if key in class_color_mapping.keys():
                mask[x, y] = class_color_mapping[key]
            elif key_reverse in class_color_mapping.keys():
                mask[x, y] = class_color_mapping[key_reverse]
            else:
                mask[x, y] = 0

    #for i in range(36):
    #    print(i, np.count_nonzero(mask==i))

    return mask


# simple function to obtain a binary mask from a dataset with multiple-classes masks
def binary_mask(mask):

    return (mask > 0).astype('float')


