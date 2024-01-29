import numpy as np
import scipy.ndimage


def prefilter_image(
        image_array: np.ndarray,
        percentile: float,
        ws_x: int,
        ws_y: int,
    ):
    """
    A prefilter to reduce coronal signals and non-zero backgrounds
    
    This is designed to be applied to each image before it goes into the
    "stack-em-all" stage, to reduce K-corona signals (and remnant F-corona
    signals) as well as clear out any constant background level.
    
    Implemented by computing a low percentile value in a neighborhood around
    each pixel, and subtracting that value from the pixel.
    
    TODO: This can be a lot better. We could try doing a bilinear fit in each
    neighborhood to better capture background level. The low-percentile, which
    is intended to determine a new zero level, could be replaced by checking
    the distribution of pixel values in the neighborhood for a Gaussian peak,
    indicating noise clustered around the "true" zero value. (This depends on
    most samples being empty space and not stars, etc.)

    Parameters
    ----------
    image_array : ``np.ndarray``
        The input image to be prefiltered
    percentile : ``float``
        The percentile value to use in each neighborhood, between 0 and 100
    ws_x : ``int``
        The size of the window in the x direction. Should be odd.
    ws_y : ``int``
        The size of the window in the y direction. Should be odd.
    
    Returns
    -------
    filtered : np.ndarray
        The image after the prefilter has been applied.
    """
    bg_estimate = scipy.ndimage.percentile_filter(
        image_array,
        percentile,
        (ws_x, ws_y),
        mode='constant',
        cval=np.inf,
    )
    
    filtered = image_array - bg_estimate
    
    return filtered