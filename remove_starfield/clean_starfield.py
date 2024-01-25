import warnings

import dask_image.ndfilters
import dask.array as da
from dask.distributed import Client
import numpy as np


def clean_map(map, percentile=1, ws=135):
    """Post-processes a starfield estimate
    
    This function subtracts a low percentile calculated in a sliding window, to
    remove any background haze that remains after the stacking. This helps
    ensure that "empty space" is black rather than gray.
    
    This function uses dask to parallelize the sliding-window-percentile
    calculation.

    Parameters
    ----------
    map : ``np.ndarray``
        The star map to be filtered
    percentile : ``float``, optional
        The percentile to subtract
    ws : ``int``, optional
        The size of the sliding window in pixels

    Returns
    -------
    cleaned map
        The cleaned map
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', 'invalid value encountered in true_divide',
            RuntimeWarning)
        client = Client(threads_per_worker=1)
        try:
            map = da.from_array(map, chunks=(-1, 500))
            map[np.isnan(map)] = np.inf
            filtered = dask_image.ndfilters.percentile_filter(
                map, percentile, ws, mode='wrap')
            return (map - filtered).compute()
        finally:
            client.shutdown()