import abc
from collections.abc import Iterable
import warnings

import numpy as np
import scipy.optimize
import scipy.stats


class StackReducer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reduce_strip(self, strip: np.ndarray) -> np.ndarray:
        """Method to reduce a section of the stack-of-reprojected-images
        
        The stack-of-reprojected-images is ``(n_input_image x ny x nx)``. In
        the reduction stage, to reduce parallelism overhead, this data cube is
        divided along the middle axis to form strips that are ``(n_input_image
        x nx)``. This method receives one of those strips and reduces along the
        first dimension. This method may return a size-``nx`` array, containing
        one reduced value for each of the nx pixels along this strip. This
        method may also return an array of shape ``(n_outputs x nx)``, where
        ``n_outputs`` is selected by this method. In this case, ``n_outputs``
        all-sky maps will be produced. This may be useful when searching for
        the correct parameters for this reduction operation, to produce outputs
        at multiple parameter values during the same pass through all the input
        data. (For example, for a percentile-based reduction, computing one
        percentile value, or computing ten percentiles in the same numpy call,
        are equally slow, so this parameter space can be explored almost for
        free.)
        
        Many values in the data slice will be NaN, indicating that a given
        input image did not contribute to a given pixel, and these NaNs should
        be ignored. When all ``n_input_image`` values for a given position
        along the second axis, NaN should be returned for that pixel.
        
        Instances of this class or subclasses must be pickleable.

        Parameters
        ----------
        strip : ``np.ndarray``
            The input array of shape (n_input_image x nx), containing all
            sample values from all input images for each pixel along a single
            horizontal slice through the all-sky map.

        Returns
        -------
        reduced_strip : np.ndarray
            The output array of shape ``(nx,)`` or ``(n_outputs x nx)``
        """


class PercentileReducer(StackReducer):
    """A `StackReducer` that calculates a percentile value at each pixel"""
    def __init__(self, percentiles: float | Iterable):
        """Configures this PercentileReducer

        Parameters
        ----------
        percentiles : ``float`` or ``Iterable``
            The percentile values to calculate. Can be one value, to produce
            one all-sky map, or multiple values, to produce multiple all-sky
            maps, one using each percentile value.
        """
        self.percentiles = percentiles
    
    def reduce_strip(self, strip):
        return np.nanpercentile(strip, self.percentiles, axis=0)


class GaussianReducer(StackReducer):
    def __init__(self, n_sigma=3):
        self.n_sigma = n_sigma
    
    def reduce_strip(self, strip):
        output = np.empty(strip.shape[1], dtype=strip.dtype)
        for i in range(len(output)):
            output[i] = self._reduce_pixel(strip[:, i])
        return output
    
    @classmethod
    def _gaussian(cls, x, x0, sigma, A):
        return A * np.exp(-(x - x0)**2 / 2 / sigma**2)
    
    def _reduce_pixel(self, sequence):
        min_size = 50
        sequence = sequence[np.isfinite(sequence)]
        if len(sequence) < min_size:
            return np.nan
        while True:
            m = np.mean(sequence)
            std = np.std(sequence)
            f = np.abs(sequence - m) < self.n_sigma * std
            if np.sum(f) <= min_size:
                return np.nan
            if np.all(f):
                break
            sequence = sequence[f]
        nbins = len(sequence) // 5
        nbins = min(nbins, 50)
        histogram, bin_edges = np.histogram(sequence, bins=nbins)
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        with np.errstate(divide='ignore'):
            sigma = 1/np.sqrt(histogram)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore',
                    message=".*Covariance of the parameters could not be "
                            "estimated.*")
                popt, pcov = scipy.optimize.curve_fit(
                    self._gaussian,
                    bin_centers, histogram,
                    [bin_centers[np.argmax(histogram)],
                    (bin_centers[-1] - bin_centers[0]) / nbins * 3,
                    np.max(histogram)],
                    sigma=sigma,
                    maxfev=4000)
            if popt[1] > 5 * (bin_centers[-1] - bin_centers[0]):
                # This is probably a bad fit
                return -np.inf
            return popt[0]
        except RuntimeError:
            return np.inf


class SkewGaussianReducer(StackReducer):
    def __init__(self, n_sigma=3):
        self.n_sigma = n_sigma
    
    def reduce_strip(self, strip):
        output = np.empty(strip.shape[1], dtype=strip.dtype)
        for i in range(len(output)):
            output[i] = self._reduce_pixel(strip[:, i])
        return output
    
    def _reduce_pixel(self, sequence):
        min_size = 50
        sequence = sequence[np.isfinite(sequence)]
        if len(sequence) < min_size:
            return np.nan
        while True:
            m = np.mean(sequence)
            std = np.std(sequence)
            f = np.abs(sequence - m) < self.n_sigma * std
            if np.sum(f) <= min_size:
                return np.nan
            if np.all(f):
                break
            sequence = sequence[f]
        nbins = len(sequence) // 5
        nbins = min(nbins, 50)
        a, loc, scale = scipy.stats.skewnorm.fit(sequence)
        if scale > 5 * (np.ptp(sequence)):
            # This is probably a bad fit
            return -np.inf
        return loc