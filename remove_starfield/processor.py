from astropy.io import fits
from astropy.wcs import WCS
import numpy as np


class ImageProcessor():
    default_wcs_key = ' '
    
    def load_image(self,
                   filename: str) -> (np.ndarray, WCS):
        """Loads an image from a given filename

        Parameters
        ----------
        filename : ``str``
            The file to load
        
        Returns
        -------
        image : ``np.ndarray``
            The image data
        hdr : ``fits.Header``
            The FITS header
        wcs : ``WCS``
            The WCS describing the image
        """
        with fits.open(filename) as hdul:
            hdu = 1 if hdul[0].data is None else 0
            hdr = hdul[hdu].header
            wcs = WCS(hdr, hdul, key=self.default_wcs_key)
            data = hdul[hdu].data
        return data, hdr, wcs
        
    def preprocess_image(self,
                         image: np.ndarray,
                         hdr: fits.Header,
                         wcs: WCS,
                         filename: str) -> np.ndarray:
        """Processes an image array before it is reprojected and stacked

        Parameters
        ----------
        image : ``np.ndarray``
            The image array
        hdr : ``fits.Header``
            The header from the FITS file
        wcs : ``WCS``
            The WCS describing the image data
        filename : ``str``
            The file path the image was loaded from

        Returns
        -------
        image : ``np.ndarray``
            The processed image array
        wcs : ``WCS``
            The processed WCS
        """
        return image, wcs
        
    def postprocess_image(self,
                         image: np.ndarray,
                         hdr: fits.Header,
                         wcs: WCS,
                         filename: str) -> np.ndarray:
        """
        Processes an image array after it is reprojected, before being stacked

        Parameters
        ----------
        image : ``np.ndarray``
            The image array
        hdr : ``fits.Header``
            The header from the FITS file
        wcs : ``WCS``
            The WCS describing the image data (post-reprojection)
        filename : ``str``
            The file path the image was loaded from

        Returns
        -------
        image : ``np.ndarray``
            The processed image array
        """
        return image