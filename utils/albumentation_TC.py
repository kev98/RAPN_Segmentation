# This script is used to set up the augmentation process for the training of the multiclass
# segmentation; the augmentations that are generally applied (each one with a probability going from 0.5 to 0.9 - to
# introduce more variability and to fasten the training process - are: horizontal flip, rotation, gaussian noise,
# median blur, random brightness contrast, motion blur, random fog, random Gamma, CLAHE and HUE Saturation values modification.
#
import albumentations as albu

def get_training_augmentation():
    train_transform = [

        #albu.HorizontalFlip(p=1),
        albu.Rotate(limit=60, p = 1),
        albu.GaussNoise(p=1),
        albu.MedianBlur(p=1),
        albu.RandomBrightnessContrast(p=1),
        albu.MotionBlur(blur_limit=3, p=1),
        albu.Blur(blur_limit=7, p=1),
        albu.MultiplicativeNoise(multiplier=(0.5, 1.5), elementwise=True, per_channel=True, p=1),
        albu.RandomFog(p=0.8),
        albu.HueSaturationValue(p=0.9),
       # albu.RandomResizedCrop(p=0.8, height=416, width=416),
    ]

    return albu.OneOf(train_transform, p=1)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
      #  albu.PadIfNeeded(384, 3)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Resize(512, 512),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)