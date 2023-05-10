import numpy as np

# binary
color_mapping_2 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (0, 130, 200),      # Instruments
}

# Define the mapping of the classes to their respective color in RGB for the MULTICLASS_1 problem
# (build using class_mapping.json and config.json)
'''color_mapping_8 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (209, 25, 30),       # Force Bipolar - RED
    2: (24, 200, 59),       # Fenestrated Bipolar Forceps - GREEN
    3: (242, 14, 138),      # Prograsp Forceps - MAGENTA
    4: (66, 62, 112),       # Monopolar Curved Scissors - DARK PURPLE
    5: (130, 1, 152),       # Suction - PURPLE
    6: (159, 89, 106),      # Large Needle Driver -
    7: (92, 154, 27),        # Echography - DARK GREEN
}'''

color_mapping_8 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (30, 25, 209),       # Force Bipolar - RED
    2: (59, 200, 24),       # Fenestrated Bipolar Forceps - GREEN
    3: (138, 14, 242),      # Prograsp Forceps - MAGENTA
    4: (112, 62, 66),       # Monopolar Curved Scissors - DARK PURPLE
    5: (152, 1, 130),       # Suction - PURPLE
    6: (106, 89, 159),      # Large Needle Driver -
    7: (27, 154, 92),        # Echography - DARK GREEN
}

color_mapping_9 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (209, 25, 30),       # Force Bipolar - RED
    2: (24, 200, 59),       # Fenestrated Bipolar Forceps - GREEN
    3: (242, 14, 138),      # Prograsp Forceps - MAGENTA
    4: (66, 62, 112),       # Monopolar Curved Scissors - DARK PURPLE
    5: (130, 1, 152),       # Suction - PURPLE
    6: (159, 89, 106),      # Large Needle Driver -
    7: (92, 154, 27),       # Echography - DARK GREEN
    8: (255, 255, 255),     # Other instruments - WHITE
}

color_mapping_10 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (209, 25, 30),       # Force Bipolar - RED
    2: (24, 200, 59),       # Fenestrated Bipolar Forceps - GREEN
    3: (242, 14, 138),      # Prograsp Forceps - MAGENTA
    4: (66, 62, 112),       # Monopolar Curved Scissors - DARK PURPLE
    5: (130, 1, 152),       # Suction - PURPLE
    6: (159, 89, 106),      # Large Needle Driver -
    7: (92, 154, 27),       # Echography - DARK GREEN
    8: (255, 0, 0),
    9: (255, 255, 255),     # Other instruments - WHITE
}

# Define the mapping of the classes to their respective color in RGB for the MULTICLASS problem (all classes)
# with Bulldog wire class
# (build using class_mapping.json and config.json)
color_mapping_23 = {
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

# Define the mapping of the classes to their respective color in RGB for the MULTICLASS problem (all classes)
# without Bulldog wire (to be updated when we'll merged some classes)
# (build using class_mapping.json and config.json)
color_mapping_22 = {
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
    #17: (137, 11, 180),     # Bulldog wire -
    17: (6, 134, 50),       # Endobag -
    18: (47, 140, 183),     # Veriset -
    19: (140, 222, 14),     # Hemolock Clip Applier -
    20: (41, 38, 64),       # Laparoscopic Needle Driver -
    21: (255, 255, 255),    # Other instruments -
}

# Define the mapping of the classes to their respective color in RGB for the MULTICLASS problem (all classes)
# with the Hemostasis class
# (build using class_mapping.json and config.json)
color_mapping_21 = {
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
    12: (255, 0, 0),        # Hemostasis-
    13: (189, 93, 206),     # Bulldog clamp -
    14: (6, 243, 246),      # Da Vinci trocar -
    15: (92, 154, 27),      # Echography - DARK GREEN
    16: (179, 254, 6),      # Laparoscopic Fenestrated Forceps -
    17: (6, 134, 50),       # Endobag -
    18: (140, 222, 14),     # Hemolock Clip Applier -
    19: (41, 38, 64),       # Laparoscopic Needle Driver -
    20: (255, 255, 255),    # Other instruments -
}

color_mapping_32 = {
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
    12: (255, 0, 0),        # Hemostasis-
    13: (189, 93, 206),     # Bulldog clamp -
    14: (6, 243, 246),      # Da Vinci trocar -
    15: (92, 154, 27),      # Echography - DARK GREEN
    16: (179, 254, 6),      # Laparoscopic Fenestrated Forceps -
    17: (180, 11, 137),     # Bulldog wire -
    18: (6, 134, 50),       # Endobag -
    19: (140, 222, 14),     # Hemolock Clip Applier -
    20: (41, 38, 64),       # Laparoscopic Needle Driver -
    21: (147, 199, 106),     # Airseal Trocar -
    22: (8, 201, 90),       # Endobag wire -
    23: (251, 98, 107),     # Endobag Specimen Retriever -
    24: (205, 133, 25),     # Laparoscopic Clip Applier -  
    25: (100, 156, 91),     # Drain -
    26: (211, 44, 115),     # Foam -
    27: (123, 208, 238),    # Metal clip -
    28: (68, 162, 66),      # Surgical_Glove_Tip -
    29: (192, 122, 96),     # Foam Extruder -
    30: (172, 199, 12),     # Laparoscopic Scissors -
    31: (233, 182, 202),    # Assistant Trocar -
}


# Define the mapping of the classes to their respective color in RGB for the MULTICLASS problem (all classes)
# with the Hemostasis class and w/o Prograsp (merged to Cadiere)
# (build using class_mapping.json and config.json)
'''color_mapping_20 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (66, 62, 112),       # Monopolar Curved Scissors - DARK PURPLE
    2: (209, 25, 30),       # Force Bipolar - RED
    3: (159, 89, 106),      # Large Needle Driver -
    4: (130, 1, 152),       # Suction - PURPLE
    5: (104, 135, 233),     # Suture wire -
    6: (176, 82, 25),       # Hemolock Clip -
    7: (24, 200, 59),       # Fenestrated Bipolar Forceps - GREEN
    8: (15, 59, 240),       # Suture needle -
    9: (132, 8, 131),       # Vessel Loop -
    10: (91, 240, 244),     # Cadiere Forceps -
    11: (255, 0, 0),        # Hemostasis-
    12: (189, 93, 206),     # Bulldog clamp -
    13: (6, 243, 246),      # Da Vinci trocar -
    14: (92, 154, 27),      # Echography - DARK GREEN
    15: (179, 254, 6),      # Laparoscopic Fenestrated Forceps -
    16: (6, 134, 50),       # Endobag -
    17: (140, 222, 14),     # Hemolock Clip Applier -
    18: (41, 38, 64),       # Laparoscopic Needle Driver -
    19: (255, 255, 255),    # Other instruments -
}'''

# Define the mapping of the classes to their respective color in RGB for the MULTICLASS problem (all classes)
# with the Hemostasis class and w/o Prograsp (merged to Cadiere), and with 'Laparoscopic Instruments' class
# (build using class_mapping.json and config.json)
'''color_mapping_19 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (66, 62, 112),       # Monopolar Curved Scissors - DARK PURPLE
    2: (209, 25, 30),       # Force Bipolar - RED
    3: (159, 89, 106),      # Large Needle Driver -
    4: (130, 1, 152),       # Suction - PURPLE
    5: (104, 135, 233),     # Suture wire -
    6: (176, 82, 25),       # Hemolock Clip -
    7: (24, 200, 59),       # Fenestrated Bipolar Forceps - GREEN
    8: (15, 59, 240),       # Suture needle -
    9: (132, 8, 131),       # Vessel Loop -
    10: (91, 240, 244),     # Cadiere Forceps -
    11: (255, 0, 0),        # Hemostasis-
    12: (189, 93, 206),     # Bulldog clamp -
    13: (6, 243, 246),      # Da Vinci trocar -
    14: (92, 154, 27),      # Echography - DARK GREEN
    15: (179, 254, 6),      # Laparoscopic Instruments -
    16: (6, 134, 50),       # Endobag -
    17: (140, 222, 14),     # Hemolock Clip Applier -
    18: (255, 255, 255),    # Other instruments -
}'''

color_mapping_19 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (112, 62, 66),       # Monopolar Curved Scissors - DARK PURPLE
    2: (30, 25, 209),       # Force Bipolar - RED
    3: (106, 89, 159),      # Large Needle Driver -
    4: (152, 1, 130),       # Suction - PURPLE
    5: (233, 135, 104),     # Suture wire -
    6: (25, 82, 176),       # Hemolock Clip -
    7: (59, 200, 24),       # Fenestrated Bipolar Forceps - GREEN
    8: (240, 59, 15),       # Suture needle -
    9: (131, 8, 132),       # Vessel Loop -
    10: (244, 240, 91),     # Cadiere Forceps -
    11: (0, 0, 255),        # Hemostasis-
    12: (206, 93, 189),     # Bulldog clamp -
    13: (246, 243, 6),      # Da Vinci trocar -
    14: (27, 154, 92),      # Echography - DARK GREEN
    15: (6, 254, 179),      # Laparoscopic Instruments -
    16: (50, 134, 6),       # Endobag -
    17: (14, 222, 140),     # Hemolock Clip Applier -
    18: (255, 255, 255),    # Other instruments -
}

def class2color(mask, n):
    """
    Function which generates a colored mask based on the input class value mask
    :param mask: A mask where each class has its own integer value
           n: number of the classes
    :return: A mask where each class has its own color
    """
    if n == 2:
        color_mapping = color_mapping_2
    elif n == 8:
        color_mapping = color_mapping_8
    elif n == 9:
        color_mapping = color_mapping_9
    elif n == 10:
        color_mapping = color_mapping_10
    elif n == 19:
        color_mapping = color_mapping_19
    elif n == 20:
        color_mapping = color_mapping_20
    elif n == 21:
        color_mapping = color_mapping_21
    elif n == 22:
        color_mapping = color_mapping_22
    elif n == 23:
        color_mapping = color_mapping_23
    elif n == 32:
        color_mapping = color_mapping_32
    else:
        return None

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


