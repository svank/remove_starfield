from dataclasses import dataclass

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from . import utils


@dataclass
class ImageHolder():
    """
    Wrapper class to hold image, WCS, and any other data
    
    Implementations of `ImageProcessor` may attach additional information as
    attributes of `ImageHolder` instances to carry necessary information
    through the load -> preprocess -> postprocess chain.
    """
    data: np.ndarray
    wcs: WCS
    meta: dict | fits.Header


class ImageProcessor():
    """Class implementing an API for instrument-specific processing
    
    By subclassing this class and passing sub-class instances into functions
    that accept a processor, the user can implement processing "hooks"
    containing any custom processing that a data set requires.
    
    When an instance of `ImageProcessor` or a subclass is passed to
    `build_starfield_estimate`, each of the input images will be loaded via
    `load_image`. If the loaded image falls within the portion of the sky map
    being assembled, `preprocess_image` will be called, where calibration,
    masking or trimming can be done. After the image is reprojected, it is
    passed to `postprocess_image` before being added to the stack of
    reprojected images.
    
    When passed to `Starfield.subtract_from_image`, the input image is loaded
    and preprocessed, but `postprocess_image` is never called. The starfield
    estimate projected into the input image's frame is passed to
    `postprocess_starfield_estimate`, and the result is subtracted from the
    input image.
    """
    def load_image(self, filename: str) -> ImageHolder:
        """Loads an image from a given filename

        Parameters
        ----------
        filename : ``str``
            The file to load
        
        Returns
        -------
        image_holder : `ImageHolder`
            An `ImageHolder` containing the image, its WCS, and any additional
            information that should be stored for later steps
        """
        with fits.open(filename) as hdul:
            image, wcs, header = utils.find_data_and_celestial_wcs(
                hdul, data=True, wcs=True, header=True)
        return ImageHolder(image, wcs, header)
        
    def preprocess_image(self, image_holder: ImageHolder) -> ImageHolder:
        """Processes an image array before it is reprojected and stacked

        Parameters
        ----------
        image_holder : `ImageHolder`
            The `ImageHolder` returned by a corresponding `load_image` call

        Returns
        -------
        image_holder : `ImageHolder`
            The `ImageHolder` after all adjustments have been made, including
            processing of the image array and modifications to the WCS
        """
        return image_holder
        
    def postprocess_image(self,
                          processed_image: np.ndarray,
                          processed_wcs: WCS,
                          image_holder: ImageHolder) -> np.ndarray:
        """
        Processes an image array after it is reprojected, before being stacked.

        Parameters
        ----------
        processed_image : ``np.ndarray``
            The reprojected image
        processed_wcs : ``WCS``
            The WCS describing the reprojected image
        image_holder : `ImageHolder`
            The `ImageHolder` of the corresponding input image

        Returns
        -------
        image : ``np.ndarray``
            The post-processed image array
        """
        return processed_image
    
    def postprocess_starfield_estimate(
            self,
            starfield_estimate: np.ndarray,
            input_image_holder: ImageHolder) -> np.ndarray:
        """
        Post-processes a starfield estimate before subtracting it from an image

        Parameters
        ----------
        starfield_estimate : ``np.ndarray``
            The starfield estimate for this image
        input_image_holder : `ImageHolder`
            The `ImageHolder` for the input image corresponding to this
            starfield estimate

        Returns
        -------
        starfield_estimate : ``np.ndarray``
            The processed starfield estimate, ready to be subtracted from the
            input image
        """
        return starfield_estimate