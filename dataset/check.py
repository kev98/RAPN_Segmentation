import cv2
import glob
import numpy as np
import os
from scipy.spatial import distance
from itertools import combinations


def main():
    # Specify the dataset folder
    source_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/masks"
    #source_folder = "/Volumes/ORSI/Kevin/Dataset_RAPN_20procedures/train/masks"
    #masks = glob.glob(source_folder + '/*/*.png')
    #dir = ['RAPN38', 'RAPN7', 'RAPN104', 'RAPN12', 'RAPN115', 'RAPN39', 'RAPN34']
    '''dir = ['RAPN91', 'RAPN20', 'RAPN96', 'RAPN102', 'RAPN47', 'RAPN41', 'RAPN87', 'RAPN92', 'RAPN81', 'RAPN95',
            'RAPN28', 'RAPN19', 'RAPN50', 'RAPN89', 'RAPN48', 'RAPN31', 'RAPN99', 'RAPN80', 'RAPN45', 'RAPN108']
    len_dir = {'RAPN91': 449,
               'RAPN20': 750,
               'RAPN96': 232,
               'RAPN102': 312,
               'RAPN47': 269,
               'RAPN41': 407,
               'RAPN87': 249,
               'RAPN92': 338,
               'RAPN81': 333,
               'RAPN95': 366,
               'RAPN28': 227,
               'RAPN19': 472,
               'RAPN50': 469,
               'RAPN89': 384,
               'RAPN48': 295,
               'RAPN31': 491,
               'RAPN99': 357,
               'RAPN80': 341,
               'RAPN45': 174,
               'RAPN108': 153}
    '''
    dir = ['RAPN102', 'RAPN41', 'RAPN87', 'RAPN92', 'RAPN81', 'RAPN95', 'RAPN28', 'RAPN19', 'RAPN89', 'RAPN31',
           'RAPN99', 'RAPN80', 'RAPN45', 'RAPN108', 'RAPN30', 'RAPN79', 'RAPN98', 'RAPN109', 'RAPN76', 'RAPN60']
    len_dir = {'RAPN102': 312,
               'RAPN41': 407,
               'RAPN87': 249,
               'RAPN92': 338,
               'RAPN81': 333,
               'RAPN95': 366,
               'RAPN28': 227,
               'RAPN19': 472,
               'RAPN89': 384,
               'RAPN31': 491,
               'RAPN99': 357,
               'RAPN80': 341,
               'RAPN45': 174,
               'RAPN108': 153,
               'RAPN30': 371,
               'RAPN79': 268,
               'RAPN98': 186,
               'RAPN109': 109,
               'RAPN76': 248,
               'RAPN60': 241}

    comb10 = list(combinations(dir, 10))
    comb11 = list(combinations(dir, 11))
    comb12 = list(combinations(dir, 12))
    comb13 = list(combinations(dir, 13))
    comb14 = list(combinations(dir, 14))
    comb15 = list(combinations(dir, 15))

    print(len(comb10), len(comb11), len(comb12), len(comb13), len(comb14), len(comb15))

    #masks.sort()

    # class distribution among the RAPN100 Dataset images, previously computed
    class_distribution = [30261, 1360, 762, 1874, 11, 61, 784, 117, 217, 5665, 18, 15239, 1500, 536, 7717, 352, 99, 936,
                          470, 30, 12650, 1, 0, 60, 22938, 4464, 0, 6, 12130, 37, 5109, 8035, 712, 3035, 0, 1031, 1093,
                          22, 244, 31]
    class_distribution = [c/30438 for c in class_distribution]

    print(class_distribution)

    # ordered list of all the classes
    classes_list =['Background', 'Bulldog clamp', 'Bulldog wire', 'Cadiere Forceps', 'Catheter',
     'Drain', 'Endobag', 'Endobag specimen retriever', 'Endobag wire', 'Fenestrated Bipolar Forceps', 'Fibrilar',
     'Force Bipolar', 'Gauze', 'Hemolock Clip Applier', 'Hemolock Clip', 'Inside Body', 'Laparoscopic Clip Applier',
     'Laparoscopic Fenestrated Forceps', 'Laparoscopic Needle Driver', 'Laparoscopic Scissors', 'Large Needle Driver',
     'Left PBP Needle Driver', 'Maryland Bipolar Forceps', 'Metal clip', 'Monopolar Curved Scissors',
     'Prograsp Forceps', 'Right PBP Needle Driver', 'Scissors', 'Suction', 'Surgical_Glove_Tip', 'Suture needle',
     'Suture wire', 'Veriset', 'Vessel Loop', 'Vessel Sealer Extend', 'Echography', 'Da Vinci trocar',
     'Assistant trocar', 'Airseal trocar', 'Foam extruder']

    #compulsory object in the test set
    compulsory_list = ['Background', 'Bulldog clamp', 'Bulldog wire',
                    'Endobag', 'Fenestrated Bipolar Forceps',
                    'Force Bipolar', 'Gauze', 'Hemolock Clip Applier', 'Hemolock Clip', 'Inside Body',
                    'Laparoscopic Fenestrated Forceps', 'Laparoscopic Needle Driver',
                    'Large Needle Driver', 'Monopolar Curved Scissors',
                    'Prograsp Forceps', 'Suction',
                    'Suture needle',
                    'Suture wire', 'Veriset', 'Vessel Loop', 'Echography', 'Da Vinci trocar']
    compulsory_index = []
    for i in compulsory_list:
        compulsory_index.append(classes_list.index(i))
    print('compulsory index', compulsory_index)

    counter = []
    for d in dir:
        count_classes = [0 for i in range(0, 40)]
        folder_mask = glob.glob(os.path.join(source_folder, d) + '/*.png')
        for mask_path in folder_mask:
            mask = cv2.imread(mask_path, 0)
            width, height = mask.shape
            flat = mask.reshape(width * height)
            # find the different colors which occur in the mask, to understand the classes present in the image
            classes = np.unique(flat)

            # loop over the image to check the classes present in it
            for c in classes:
                count_classes[c] += 1
        counter.append(count_classes)

    counter = np.array(counter)

    print(counter)


    min_distances = 1000
    best_list = []
    best_distribution = []
    best_len = 0

    for co in comb10:
        curr_len = 0
        curr_counter = []
        for d in co:
            curr_len += len_dir[d]
        if curr_len < 3600 or curr_len > 4300:
            continue
        for d in co:
            curr_counter.append(counter[dir.index(d)])

        count_classes = np.sum(curr_counter, axis=0)

        lack_class = False
        for ind in compulsory_index:
            if count_classes[ind] == 0:
                lack_class = True
        if lack_class == True:
            print(co, 'Lack class')
            continue
        print(co)

        count_classes = [i / curr_len for i in count_classes]
        dist = distance.euclidean(count_classes, class_distribution)
        if dist < min_distances:
            min_distances = dist
            best_list = co
            best_distribution = count_classes
            best_len = curr_len
            print(best_list, min_distances)

    for co in comb11:
        curr_len = 0
        curr_counter = []
        for d in co:
            curr_len += len_dir[d]
        if curr_len < 3600 or curr_len > 4300:
            continue
        for d in co:
            curr_counter.append(counter[dir.index(d)])

        count_classes = np.sum(curr_counter, axis=0)

        lack_class = False
        for ind in compulsory_index:
            if count_classes[ind] == 0:
                lack_class = True
        if lack_class == True:
            print(co, 'Lack class')
            continue
        print(co)

        count_classes = [i / curr_len for i in count_classes]
        dist = distance.euclidean(count_classes, class_distribution)
        if dist < min_distances:
            min_distances = dist
            best_list = co
            best_distribution = count_classes
            best_len = curr_len
            print(best_list, min_distances)

    for co in comb12:
        curr_len = 0
        curr_counter = []
        for d in co:
            curr_len += len_dir[d]
        if curr_len < 3600 or curr_len > 4300:
            continue
        for d in co:
            curr_counter.append(counter[dir.index(d)])

        count_classes = np.sum(curr_counter, axis=0)

        lack_class = False
        for ind in compulsory_index:
            if count_classes[ind] == 0:
                lack_class = True
        if lack_class == True:
            print(co, 'Lack class')
            continue
        print(co)

        count_classes = [i / curr_len for i in count_classes]
        dist = distance.euclidean(count_classes, class_distribution)
        if dist < min_distances:
            min_distances = dist
            best_list = co
            best_distribution = count_classes
            best_len = curr_len
            print(best_list, min_distances)

    for co in comb13:
        curr_len = 0
        curr_counter = []
        for d in co:
            curr_len += len_dir[d]
        if curr_len < 3600 or curr_len > 4300:
            continue
        for d in co:
            curr_counter.append(counter[dir.index(d)])

        count_classes = np.sum(curr_counter, axis=0)

        lack_class = False
        for ind in compulsory_index:
            if count_classes[ind] == 0:
                lack_class = True
        if lack_class == True:
            print(co, 'Lack class')
            continue
        print(co)

        count_classes = [i / curr_len for i in count_classes]
        dist = distance.euclidean(count_classes, class_distribution)
        if dist < min_distances:
            min_distances = dist
            best_list = co
            best_distribution = count_classes
            best_len = curr_len
            print(best_list, min_distances)

    for co in comb14:
        curr_len = 0
        curr_counter = []
        for d in co:
            curr_len += len_dir[d]
        if curr_len < 3600 or curr_len > 4300:
            continue
        for d in co:
            curr_counter.append(counter[dir.index(d)])

        count_classes = np.sum(curr_counter, axis=0)

        lack_class = False
        for ind in compulsory_index:
            if count_classes[ind] == 0:
                lack_class = True
        if lack_class == True:
            print(co, 'Lack class')
            continue
        print(co)

        count_classes = [i / curr_len for i in count_classes]
        dist = distance.euclidean(count_classes, class_distribution)
        if dist < min_distances:
            min_distances = dist
            best_list = co
            best_distribution = count_classes
            best_len = curr_len
            print(best_list, min_distances)

    for co in comb15:
        curr_len = 0
        curr_counter = []
        for d in co:
            curr_len += len_dir[d]
        if curr_len < 3600 or curr_len > 4300:
            continue
        for d in co:
            curr_counter.append(counter[dir.index(d)])

        count_classes = np.sum(curr_counter, axis=0)

        lack_class = False
        for ind in compulsory_index:
            if count_classes[ind] == 0:
                lack_class = True
        if lack_class == True:
            print(co, 'Lack class')
            continue
        print(co)

        count_classes = [i / curr_len for i in count_classes]
        dist = distance.euclidean(count_classes, class_distribution)
        if dist < min_distances:
            min_distances = dist
            best_list = co
            best_distribution = count_classes
            best_len = curr_len
            print(best_list, min_distances)



    print('The minimum list found is: ', best_list)
    print('The length of test set is: ', best_len)
    print("It's euclidean distance between the dataset is: ", min_distances)
    print('The best distribution is: ')

    for i in range(len(best_distribution)):
        print(classes_list[i], best_distribution[i])


if __name__ == '__main__':
    main()
