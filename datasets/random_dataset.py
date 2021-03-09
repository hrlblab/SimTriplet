import torch
import csv
import os
from PIL import Image

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_list_file,  train=True, transform=None):
        self.data_root = data_dir
        self.transform = transform
        with open(image_list_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        self.image_list = rows
        self.size = len(rows)

    def __getitem__(self, idx):
        if idx < self.size:
            origin_img_path = self.image_list[idx][0].replace('/original/', '/original_nearby_80000/original_nearby_8wan/')
            for nearby_index in range(8):
                nearby_img_path = origin_img_path.replace('.png', '_' + str(nearby_index) + '.png')
                if os.path.exists(nearby_img_path):
                    break
            origin_img = Image.open(origin_img_path)
            nearby_img = Image.open(nearby_img_path)

            if self.transform:
                origin, nearby, origin_z = self.transform(origin_img, nearby_img)
            return origin, nearby, origin_z, [0]
        else:
            raise Exception

    def __len__(self):
        return self.size
