import numpy as np

# binary
color_mapping_2 = {
    0: (0, 0, 0),           # Background - BLACK
    1: (255, 255, 255),      # Instruments
}

def class2color(mask):
    """
    Function which generates a colored mask based on the input class value mask
    :param mask: A mask where each class has its own integer value
           n: number of the classes
    :return: A mask where each class has its own color
    """

    color_mapping = color_mapping_2

    # Initialize a blank image
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])

    # Iterate over the possible class values, as defined in class_color_mapping
    for i in color_mapping:
        # Assign the color corresponding to the class value to the appropriate pixels in the blank image
        color_mask[np.where(mask == i)] = color_mapping[i]

    return color_mask