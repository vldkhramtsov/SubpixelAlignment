# SubpixelAlignment
Implementation of image alignment through phase correlation in Fourier space for pixel- and subpixel-bias.

## Dataset
The data is located in the ``data`` directory. It contains two datasets: ``saturn`` and ``sentinel2``. Each dataset consists of two ``.tiff`` images of the one region, separated in time.

## Code
The code is located in the ``code`` directory. It contains the scripts for dataset uploading (``dataset.py``), drawing (``plot.py``), and detection of the offset between images via [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation).

**Note**: if the offset is assumed to be on the order of pixels, the parameter ``upsample_factor`` should be equal to ``1`` (``saturn`` dataset case); otherwise, if the shift between images is observed on the subpixel level, the parameter ``upsample_factor`` should be more than ``1`` (typically ``3-10``; ``sentinel2`` dataset case).

The example of using the code is presented in the Jupyter Notebook (``data/example.ipynb``).

## Requirements
- numpy==1.18.5
- rasterio==1.1.5
- scikit-image==0.17.2
- scipy==1.4.1

(see requirements.txt for the more details)
