import os
from os.path import join
import rasterio as rs
from rasterio.plot import reshape_as_image as to_img

IMG_FOLDER = '../data'
CUT_OFF = 200

class Dataset:
    def __init__(self, source):
        self.source = source
    
    def read_tiff(self, filename):
        src = rs.open(filename)
        return to_img(src.read()).squeeze()

    def sample_images(self):
        if self.source == 'saturn' or self.source == 'sentinel2':
            PATH_TO_IMGS = join(IMG_FOLDER, self.source)
            images, counter = {}, 0
            for file in os.listdir(PATH_TO_IMGS):
                if file.endswith('.tiff'):
                    key = f"image_{counter}"
                    images[key] = self.read_tiff(join(PATH_TO_IMGS, file))
                    images[key] = images[key][:CUT_OFF, :CUT_OFF]
                    counter += 1
            return images
        else:
            raise ValueError("Error: source of images is not known")
