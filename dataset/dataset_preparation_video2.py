"""
Script to prepare the RAPN100 Dataset with sequence of frames (video) for the semantic segmentation.
The script create a new usable dataset only with the images useful for the Instrument Segmentation task,
cropped in the rigth way
- Step 1. Cleaning of the dataset: removal of all the "sequences" which contain only classes not useful for the segmentation
  of the instruments (like 'Outside body', 'Ecography image, 'Color bar' etc...)
- Step 2. Cropping of the images: compute the contours of the images, then find the biggest contour which
  which contains classes of interest and crop it
- Step 3. Conversion of the mask: mask are converted from RGB to one channel images, mapping each pixel to the
  corresponding class; furthermore different instances of the same class are merged into a single class
- Step 4. Group the adjacent frames in a sequence, in which the final frame is always the one for which we have
  already a manually annotated images

  P.s. In the comment of the script when I referred to RAPN100, I always mean the Dataset "filtered" created by me
       for the semantic segmentation of the instrument
"""

import os.path
import cv2
import glob
import csv


# Function to create the directory tree of the dataset, if it's necessary
def create_dataset_tree(source, dest):
    if not os.path.exists(dest):
        os.mkdir(dest)
        os.mkdir(os.path.join(dest, 'raw_images'))
        os.mkdir(os.path.join(dest, 'masks'))

    dir = os.listdir(source)
    for d in dir:
        if '.csv' not in d:
            img_path = os.path.join((dest + '/raw_images'), d)
            mask_path = os.path.join((dest + '/masks'), d)
            if not os.path.exists(img_path):
                os.mkdir(img_path)
                os.mkdir(mask_path)


# Function which crop the part of the image we are interested, and check if each image of the sequence is
# coherent with the sequence that we are creating
def check_and_crop(img, width, height):
    # compute the contours and the areas of them
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]

    if len(areas) == 0:
        return None, False

    # order the contours in descending order based on the value of their area
    index = sorted(range(len(areas)), key=lambda k: areas[k], reverse=True)
    check = False
    img_crop = None

    for i in index:
        cnt = contours[i]
        # img_c = cv2.drawContours(img, contours, max_index, color=[0,255,0], thickness=2)
        x, y, w, h = cv2.boundingRect(cnt)
        # check if there is a bounding box in the current image, coherent with the next correspondent image
        # in the sequence of the RAPN100
        if w < 0.99*width or w > 1.01*width or h < 0.99*height or h > 1.01*height:
            continue
        img_crop = img[y:y + h, x:x + w]  # crop the image
        check = True
        break
        #cv2.imshow('image', img_crop)
        #cv2.waitKey(0)

    return img_crop, check


def main():
    # Specify the input folder and the output folder
    source_folder = r"/home/kmarc/workspace/nas_private/samples2"
    #source_folder = "/Volumes/ORSI/Kevin/samples"
    dest_folder = r"/home/kmarc/workspace/nas_private/Video_Segmentation_Dataset_RAPN"
    #dest_folder = "/Volumes/ORSI/Kevin/video_dataset"
    train_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/train"
    valid_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/val"
    test_folder = r"/home/kmarc/workspace/nas_private/Segmentation_Dataset_RAPN/test"
    #train_folder = "/Volumes/ORSI/Kevin/RAPN100/train"
    #valid_folder = "/Volumes/ORSI/Kevin/RAPN100/val"
    #test_folder =  "/Volumes/ORSI/Kevin/RAPN100/test"
    create_dataset_tree(source_folder, dest_folder)

    train_proc = os.listdir(os.path.join(train_folder, 'raw_images'))
    valid_proc = os.listdir(os.path.join(valid_folder, 'raw_images'))
    test_proc = os.listdir(os.path.join(test_folder, 'raw_images'))
    print('Train procedures: ', train_proc)
    print('Validation procedures: ', valid_proc)
    print('Test procedures: ', test_proc)

    list_proc = {'train': train_proc, 'valid': valid_proc, 'test':test_proc}

    # loop over the train, validation and test set procedures
    for set, l in list_proc.items():
        for p in l:

            # Code for the creation of sequences for the proceure p

            count_seq = 0
            # .csv file with the informatipn of the frames sampled at 1s
            csv_path = os.path.join(source_folder, p) + '.csv'
            if not os.path.isfile(csv_path):
                continue

            with open(csv_path, 'r') as file:
                csv_reader = csv.reader(file, delimiter=',')
                # list of images of the procedure p, present in the RAPN100 filtered
                if set == 'train':
                    images = glob.glob(os.path.join(train_folder, 'raw_images', p) + '/*.png')
                    images.sort()
                    relative_images = [os.path.relpath(i, os.path.join(train_folder, 'raw_images', p)) for i in images]
                if set == 'valid':
                    images = glob.glob(os.path.join(valid_folder, 'raw_images', p) + '/*.png')
                    images.sort()
                    relative_images = [os.path.relpath(i, os.path.join(valid_folder, 'raw_images', p)) for i in images]
                if set == 'test':
                    images = glob.glob(os.path.join(test_folder, 'raw_images', p) + '/*.png')
                    images.sort()
                    relative_images = [os.path.relpath(i, os.path.join(test_folder, 'raw_images', p)) for i in images]

                relative_images = [i.split('.')[0] for i in relative_images]

                csv_reader = list(csv_reader)
                csv_reader.pop(0)
                prev_img, curr_img = '', ''
                # loop ove each frame sampled at 1s
                for row in csv_reader:
                    # code to understand what are the current and the previous images of the RAPN100 filtered
                    # (if they are in the dataset)
                    rapn_image = row[-1]
                    if rapn_image == '':
                        continue
                    count_seq += 1
                    if prev_img == '':
                        if rapn_image in relative_images:
                            prev_img = images[relative_images.index((rapn_image))]
                        else:
                            prev_img, curr_img = '', ''
                            continue
                    else:
                        if rapn_image in relative_images:
                            if curr_img == '':
                                curr_img = images[relative_images.index((rapn_image))]
                            else:
                                prev_img = curr_img
                                curr_img = images[relative_images.index((rapn_image))]
                        else:
                            prev_img, curr_img = '', ''
                            continue

                    # if two consecutive frame of the RAPN100 are taken, and they have very similar width and height
                    # (so they are consistent) then create a sequence with all the frames (20) sampled at 1s
                    # between those frames
                    if prev_img != '' and curr_img != '':
                        # open the previous and the current image of the RAPN Segmentation Dataset
                        prev_image = cv2.cvtColor(cv2.imread(prev_img), cv2.COLOR_BGR2RGB)
                        curr_image = cv2.cvtColor(cv2.imread(curr_img), cv2.COLOR_BGR2RGB)

                        # retrieve dimensions of the previous and the current images
                        p_h, p_w = prev_image.shape[0], prev_image.shape[1]
                        c_h, c_w = curr_image.shape[0], curr_image.shape[1]
                        if c_w < 0.99*p_w or c_w > 1.01*p_w or c_h < 0.99*p_h or c_h > 1.01*p_h:
                            # if the dimension of the previous and the current image are too different go ahead
                            # --> you don't have a uniform sequence
                            continue

                        n_frame = int(row[0])
                        list_frames = [frame[1].replace('_', '', 1) for frame in csv_reader[n_frame-19:n_frame]]
                        sequence = []
                        # loop over the frame between the previous and the current frame of the RAPN100
                        for frame_path in list_frames:
                            frame_path = os.path.join(source_folder, p, frame_path)
                            if not os.path.isfile(frame_path):
                                sequence = []
                                break
                            frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
                            # check if the image is similar to the "current one" of the RAPN100 dataset
                            # and crop it properly
                            image, check = check_and_crop(frame, c_w, c_h)
                            # if at least one image of the current sequence has shapes too different from the others
                            # discard the sequence and doesn't save it
                            if not check:
                                sequence = []
                                break
                            sequence.append(image)

                        # if we have a complete sequence (19+1 consecutive frames)
                        if sequence:
                            # create a new directory for the sequence
                            dir_images = os.path.join(dest_folder, 'raw_images', p, str(count_seq).zfill(3))
                            dir_masks = os.path.join(dest_folder, 'masks', p, str(count_seq).zfill(3))
                            if not os.path.exists(dir_images):
                                os.mkdir(dir_images)
                            if not os.path.exists(dir_masks):
                                os.mkdir(dir_masks)

                            # save all the images of the sequence in the directory
                            for i, im in enumerate(sequence):
                                im_dest_path = os.path.join(dir_images, str(i).zfill(2) + '.png')
                                cv2.imwrite(im_dest_path, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

                            # save the image of the RAPN100, which is the only one with a manually annotated mask
                            # and is always the last image of the sequence
                            im_dest_path = os.path.join(dest_folder, 'raw_images', p, str(count_seq).zfill(3), rapn_image + '.png')
                            cv2.imwrite(im_dest_path, cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB))
                            # open and save the correspondent manually annotated mask
                            mask_path = curr_img.replace('raw_images', 'masks')
                            mask_dest_path = os.path.join(dest_folder, 'masks', p, str(count_seq).zfill(3), rapn_image + '.png')
                            curr_mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
                            cv2.imwrite(mask_dest_path, cv2.cvtColor(curr_mask, cv2.COLOR_BGR2RGB))

                            # Notify in output the creation of the sequence
                            print('Sequence {} of procedure {} created'.format(str(count_seq).zfill(3), p))
                            print("The following frames have been taken: ", list_frames)


if __name__ == '__main__':
    main()
