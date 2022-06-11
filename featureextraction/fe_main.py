import torch
from torchvision import transforms
from featureextraction.solar.solar_global.networks.imageretrievalnet import extract_vectors, extract_vectors_labels, extract_vectors_orig
from featureextraction.solar.solar_global.utils.networks import load_network
from featureextraction.solar.solar_global.datasets.genericdataset import ImagesFromList, ImagesFromDataList
from dataset.DatasetUtils import ImageDataset


def extract_features_global(images, size=256):
    """Extract features from the given `search_images` and `frame_images`.

    Arguments:
        loader:         a DataLoader that loads the images
        size:           the size to resize the (shortest size of the) images to

    Returns:
        image_features: tuple containing the extracted features of the
                        `search_images` and the `frame_images`
    """
    net = load_network('resnet101-solar-best.pth')
    net.mode = 'test'

    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),  # the FromImageDataList() that is used in extract_vectors uses
        # Tensors
        transforms.Resize(size=size),
        normalize
    ])

    ms = [1, 2**(1/2), 1/2**(1/2)]

    loader = torch.utils.data.DataLoader(
        ImagesFromDataList(images, transform)
    )

    image_features = extract_vectors(net, images, size, transform, ms=ms, mode='test')

    image_features = image_features.transpose(0, 1)

    return image_features


def extract_features_alt(images, size=256, root=''):
    """Extract features from the given `search_images` and `frame_images`.

    Arguments:
        loader:         a DataLoader that loads the images
        size:           the size to resize the (shortest size of the) images to

    Returns:
        image_features: tuple containing the extracted features of the
                        `search_images` and the `frame_images`
    """
    net = load_network('resnet101-solar-best.pth')
    net.mode = 'test'

    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),  # the FromImageDataList() that is used in extract_vectors uses
                                # Tensors
        transforms.Resize(size=size),
        normalize
    ])

    ms = [1, 2**(1/2), 1/2**(1/2)]

    # loader = torch.utils.data.DataLoader(
    #     ImagesFromDataList(images, transform)
    # )

    image_features = extract_vectors_orig(net, root, images, size, transform, ms=ms, mode='test')

    image_features = image_features.transpose(0, 1)

    return image_features


def extract_features(image_dir, labels, size):
    net = load_network('resnet101-solar-best.pth')
    net.mode = 'test'

    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    # Do not use the ToTensor() transform, as DataLoader that is used in extract_vectors_labels
    # already reads in images as Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=size),
        normalize
    ])

    ms = [1, 2**(1/2), 1/2**(1/2)]

    image_features, image_labels, image_names = extract_vectors_labels(net, labels, image_dir, transform, ms=ms)

    image_features = image_features.transpose(0, 1)

    return image_features, image_labels, image_names
