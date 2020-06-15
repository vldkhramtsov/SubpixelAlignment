import numpy as np
from scipy.ndimage import shift
from skimage.transform import rescale, downscale_local_mean
from skimage.util import img_as_ubyte

class PhaseCorrelation:
    """
    PhaseCorrelation class for image alignment
    by correlation in Fourier frequency space
   
    Parameters
    ----------
    src_image : ndarray
        Reference image.
    target_image : ndarray
        Image to align with the same dimensionality as src_image.
    upsample_factor : float, optional
        Upsampling factor. If upsample_factor > 1, then images
        will be upsampled on the specified factor. 
        Default is upsample_factor = 1 (no upsampling)
    """

    def __init__(self, src_image, target_image, upscale_factor=1):
        self.src_image = np.copy(src_image)
        self.target_image = np.copy(target_image)
        self.upscale_factor = upscale_factor
    
    def fourier_space(self, image):
        """
        FFT transform of image.
    
        Parameters
        ----------
        image : ndarray
            Image to transform.
        
        Returns
        -------
        image_freq : ndarray
            Transformed image.
        """
        
        image = np.array(image, dtype=np.complex128)
        image_freq = np.fft.fftn(image)
        return image_freq

    def resample(self, image, scale):
        image = np.array(image, dtype=np.uint8)
        if scale > 1:
            return img_as_ubyte(rescale(image, scale=scale, order=1, multichannel=(image.ndim == 3)))
        else:
            factors = np.ones(image.ndim, dtype=np.int)*int(1/scale)
            if image.ndim == 3: factors[-1] = 1
            else: factors[-1] = int(1/scale)
            return np.array(downscale_local_mean(image, factors=tuple(factors)), dtype=np.uint8)

    def cross_correlation(self):
        """
        Calculate the cross-power spectrum by taking the complex conjugate.
    
        Returns
        -------
        np.fft.ifftn(image_product) : ndarray
        shape : tuple of int
            Normalized cross-correlation by applying the inverse Fourier transform.
            Shape of input images.
        """

        assert self.src_image.shape == self.target_image.shape
        assert self.src_image.ndim == 2 or self.src_image.ndim == 3
        assert self.target_image.ndim == 2 or self.target_image.ndim == 3
        assert self.upscale_factor > 0 and self.upscale_factor < 100

        if self.upscale_factor != 1:
            src = self.resample(self.src_image, self.upscale_factor)
            target = self.resample(self.target_image, self.upscale_factor)
        else:
            src = self.src_image
            target = self.target_image
        
        if src.ndim == 3: src = src[:, :, 0]
        if target.ndim == 3: target = target[:, :, 0]

        shape = src.shape
        src_f = self.fourier_space(src)
        target_f = self.fourier_space(target)

        image_product = src_f * target_f.conj()
        return np.fft.ifftn(image_product), shape
    
    def get_shift(self):
        """
        Get the location of cross correlation maximum
    
        Returns
        -------
        shifts : ndarray
            Shifts of target image with respect to the source image (in pixels).
        """
        cross_corr, shape = self.cross_correlation()

        maxima = np.unravel_index(np.argmax(np.abs(cross_corr)),
                                  cross_corr.shape)
        midpoints = np.array([np.fix(axis_shape / 2) for axis_shape in shape])
        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
        return shifts

    def transform(self):
        """
        Transform the target image
    
        Returns
        -------
        shifted_image : ndarray
            Target image after phase correlation shifting.
        """
        shift_pixels = self.get_shift()
        print(f"Detected shift: {shift_pixels / self.upscale_factor}")
        target = self.resample(self.target_image, self.upscale_factor)
        
        if target.ndim == 3:
            shifted_image = np.zeros(target.shape)
            for axis in range(target.shape[-1]):
                shifted_image[:, :, axis] += shift(target[:, :, axis].squeeze(),
                                                   shift_pixels,
                                                   cval=np.mean(target[:, :, axis]))
        else:
            shifted_image = shift(target, shift_pixels, cval=np.mean(target))
        
        if self.upscale_factor != 1:
            shifted_image = self.resample(shifted_image, 1/self.upscale_factor)
        return shifted_image
