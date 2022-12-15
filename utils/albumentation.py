# This script is used to set up the augmentation process for the training of the multiclass
# segmentation; the augmentations are generally applied (each one with a probability going from 0.5 to 0.9 - to
# introduce more variability and to fasten the training process


import albumentations as albu


# AUGMENTATION FOR TRAINING DATA
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=1),
        albu.Rotate(limit=35, p = 1),
        albu.GaussNoise(p=1),
        albu.MedianBlur(p=1),
        albu.RandomBrightnessContrast(p=1),
        albu.MotionBlur(blur_limit=3, p=1),
        albu.Blur(blur_limit=7, p=1),
        albu.MultiplicativeNoise(multiplier=(0.5, 1.5), elementwise=True, per_channel=True, p=1),
        albu.RandomFog(p=0.8),
        albu.HueSaturationValue(p=0.9),
    ]

    return albu.OneOf(train_transform, p=0.3)


# AUGMENTATION FOR VALIDATION DATA
def get_validation_augmentation():
    test_transform = [
       # albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


# TRANSFORMING THE IMAGE INTO A TENSOR OF FLOAT
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


# PREPROCESSING FUNCTION
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


# PREPROCESSING FUNCTION FOR PESUDOLABELING AND IN GENERAL FOR FINAL INFERENCE WITHOUT MASK
def get_inference_preprocessing(preprocessing_fn):
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
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)
