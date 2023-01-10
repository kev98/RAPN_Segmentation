import numpy as np

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

# Define the mapping of the classes to their respective color in RGB for the MULTICLASS_1 problem
# (build using class_mapping.json and config.json)
'''color_mapping = {
    0: (0, 0, 0),           # Background - BLACK
    1: (209, 25, 30),       # Force Bipolar - RED
    2: (24, 200, 59),       # Fenestrated Bipolar Forceps - GREEN
    3: (242, 14, 138),      # Prograsp Forceps - MAGENTA
    4: (66, 62, 112),       # Monopolar Curved Scissors - DARK PURPLE
    5: (130, 1, 152),       # Suction - PURPLE
    6: (159, 89, 106),      # Large Needle Driver -
    7: (92, 154, 27)        # Echography - DARK GREEN
}'''

# Define the mapping of the classes to their respective color in RGB for the MULTICLASS problem (all classes)
# (build using class_mapping.json and config.json)
color_mapping = {
    0: (0, 0, 0),           # Background - BLACK
    1: (66, 62, 112),       # Monopolar Curved Scissors - DARK PURPLE
    2: (209, 25, 30),       # Force Bipolar - RED
    3: (159, 89, 106),      # Large Needle Driver -
    4: (130, 1, 152),       # Suction - PURPLE
    5: (104, 135, 233),     # Suture wire -
    6: (176, 82, 25),       # Hemolock Clip -
    7: (24, 200, 59),       # Fenestrated Bipolar Forceps - GREEN
    8: (15, 59, 240),       # Suture needle
    9: (242, 14, 138),      # Prograsp Forceps - MAGENTA
    10: (132, 8, 131),      # Vessel Loop -
    11: (91, 240, 244),     # Cadiere Forceps -
    12: (61, 102, 44),      # Gauze -
    13: (189, 93, 206),     # Bulldog clamp -
    14: (6, 243, 246),      # Da Vinci trocar -
    15: (92, 154, 27),      # Echography - DARK GREEN
    16: (179, 254, 6),      # Laparoscopic Fenestrated Forceps -
    17: (137, 11, 180),     # Bulldog wire -
    18: (6, 134, 50),       # Endobag -
    19: (47, 140, 183),     # Veriset -
    20: (140, 222, 14),     # Hemolock Clip Applier -
    21: (41, 38, 64),       # Laparoscopic Needle Driver -
    22: (255, 255, 255),    # Other instruments -
}


def class2color(mask):
    """
    Function which generates a colored mask based on the input class value mask
    :param mask: A mask where each class has its own integer value
    :return: A mask where each class has its own color
    """

    # Initialize a blank image
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])

    # Iterate over the possible class values, as defined in class_color_mapping
    for i in color_mapping:
        # Assign the color corresponding to the class value to the appropriate pixels in the blank image
        color_mask[np.where(mask == i)] = color_mapping[i]

    return color_mask


def color2class(color_mask, class_color_mapping):
    """
    Function which generates a class value mask based on the input colored mask
    :param color_mask: A mask where each class has its own color
           class_color_mapping: A dictionary which maps each color into a class
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


