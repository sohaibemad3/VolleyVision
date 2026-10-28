"""
Image transformations for the Volleyball Group Activity Recognition project.

Four transformation pipelines are provided:
1. Image classification.
2. Cropped player classification.
3. Image sequence classification.
4. Cropped player sequence classification.

All pipelines use ImageNet normalization because the models use
pretrained ResNet50 weights.
"""

from torchvision import transforms


# ImageNet statistics used by the pretrained ResNet50 backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def load_image_classification_transforms():
    """
    Transform full images for image-level classification.

    Training:
        Resize, random crop, color augmentation, horizontal flip,
        and normalization.

    Evaluation:
        Deterministic resizing and center cropping.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform


def load_cropped_player_transforms():
    """
    Transform individual cropped players for player-action classification.

    Mild augmentation helps the classifier generalize to variations
    in lighting and player appearance.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform


def load_image_sequence_transforms():
    """
    Transform frames used in image-sequence classification.

    Every frame receives the same output dimensions.
    Random spatial transformations are intentionally omitted here
    to avoid introducing inconsistent transformations across frames
    when frames are processed independently.
    """
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform


def load_cropped_player_sequence_transforms():
    """
    Transform cropped player frames used in temporal player modeling.

    Fixed spatial resizing preserves consistent tensor dimensions
    across players and frames. Random spatial augmentations are omitted
    because independent transformations can introduce artificial motion.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform
