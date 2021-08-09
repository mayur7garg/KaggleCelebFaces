import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence

IMG_BASE_PATH = r'../Dataset/img_align_celeba'
IMAGE_SIZE = (192, 160)
LANDMARKS = ['lefteye', 'righteye', 'nose', 'leftmouth', 'rightmouth']

class LandmarkCutoutImageIterator(Sequence):
    def __init__(self, df, batch_size, step_count, max_cutouts = 2, min_cutout_size = 12, max_cutout_size = 15):
        self.df = df
        self.batch_size = batch_size
        self.step_count = step_count
        self.max_cutouts = max_cutouts
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size
    
    def __len__(self):
        return self.step_count
    
    def __getitem__(self, idx):
        x = []
        masks = []
        y = []

        for _ in range(self.batch_size):
            row = self.df.sample()
            img = Image.open(os.path.join(IMG_BASE_PATH, row['image_id'].iloc[0]))
            w, h = img.size
            img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            img = np.array(img)/255.0

            cutout_img = img.copy()
            mask = np.ones((*IMAGE_SIZE, 1))
            cutout_count = random.randint(1, self.max_cutouts)
            cutout_features = random.sample(LANDMARKS, cutout_count)

            for feature in cutout_features:
                cutout_x = int((row[feature + '_x'].iloc[0]/w) * IMAGE_SIZE[1])
                cutout_y = int((row[feature + '_y'].iloc[0]/h) * IMAGE_SIZE[0])

                cutout_size = random.randint(self.min_cutout_size, self.max_cutout_size)
                
                mask[cutout_y - cutout_size: cutout_y + cutout_size, cutout_x - cutout_size: cutout_x + cutout_size] = 0
            
            cutout_img = cutout_img * mask

            x.append(cutout_img)
            masks.append(mask)
            y.append(img)

        return ((np.array(x), np.array(masks)), np.array(y))