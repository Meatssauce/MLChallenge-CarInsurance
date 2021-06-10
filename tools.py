import pandas as pd
from PIL import Image
import numpy as np
import os


def load_data(csv_path, image_dir, image_size, grey_scale):
    df = pd.read_csv(csv_path)

    images = []
    for file_name in df['Image_path']:
        image_path = os.path.join(image_dir, file_name)

        try:
            with Image.open(image_path) as image:
                image.thumbnail(image_size)
                if grey_scale:
                    image.convert('LA')
                image = np.asarray(image)
                images.append(image)
        except FileNotFoundError:
            image.append(np.nan)

    df['Images'] = images

    return df
