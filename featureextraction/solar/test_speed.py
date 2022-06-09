"""
Test the speed at which certain extraction methods work.
"""
import torch
from torchvision import transforms
from featureextraction.solar.solar_local.models.model import SOLAR_LOCAL
import time


def time_solar_local():
    options = {
        'soa': True,
        'soa_layers': '345',
        'train_set': 'liberty'
    }

    grayscale = transforms.Grayscale(num_output_channels=1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        grayscale
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    solar_local = SOLAR_LOCAL(soa=options['soa'], soa_layers=options['soa_layers'])

    model_weight_path = '/home/aron/repos/SOLAR/solar_local/weights/local-solar-345-liberty.pth'
    state_dict = torch.load(model_weight_path)
    solar_local.load_state_dict(state_dict)
    solar_local = solar_local.to(device)
    solar_local.eval()

    # extract descriptors here

    inputs = torch.rand(1, 1, 512, 512)
    start_time = time.time()
    with torch.no_grad():
        patches = inputs.to(device)
        descrs = solar_local(patches)
    end_time = time.time()
    print(f'Time spent: {end_time - start_time} seconds')

    return descrs


def main():
    time_solar_local()


if __name__ == '__main__':
    main()
