import abc
from collections.abc import Iterable
import copy
from dataclasses import dataclass
from itertools import repeat
from math import ceil, floor
import multiprocessing
import random

from astropy.io import fits
import astropy.visualization.wcsaxes
import astropy.units as u
from astropy.wcs import WCS
import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import reproject
import scipy.optimize
from tqdm.auto import tqdm
import warnings

from . import utils


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
        wcs : ``WCS``
            The WCS describing the image
        """
        with fits.open(filename) as hdul:
            hdu = 1 if hdul[0].data is None else 0
            hdr = hdul[hdu].header
            wcs = WCS(hdr, hdul, key=self.default_wcs_key)
            data = hdul[hdu].data
        return data, wcs
        
    def preprocess_image(self,
                         image: np.ndarray,
                         wcs: WCS,
                         filename: str) -> np.ndarray:
        """Processes an image array before it is reprojected and stacked

        Parameters
        ----------
        image : ``np.ndarray``
            The image array
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
                         wcs: WCS,
                         filename: str) -> np.ndarray:
        """
        Processes an image array after it is reprojected, before being stacked

        Parameters
        ----------
        image : ``np.ndarray``
            The image array
        wcs : ``WCS``
            The WCS describing the image data
        filename : ``str``
            The file path the image was loaded from

        Returns
        -------
        image : ``np.ndarray``
            The processed image array
        """
        return image


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
        self.percentile = percentiles
    
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
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])
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
            return popt[0]
        except RuntimeError:
            return np.inf


def build_starfield_estimate(
        files: Iterable[str],
        frame_count: bool=False,
        attribution: bool=False,
        processor: ImageProcessor=ImageProcessor(),
        reducer: "StackReducer"=GaussianReducer(),
        ra_bounds: Iterable[float]=None,
        dec_bounds: Iterable[float]=None,
        stack_all: bool=False,
        shuffle: bool=True):
    """Generate a starfield estimate from a set of images
    
    This is generally a slow, high-memory-use function, as each image must be
    reprojected into the frame of the output all-sky map, and all reprojected
    images must be held in memory to compute the low-percentile value at each
    pixel. To contain the memory usage, the output map is divided into chunks
    which are each computed separately. This should generally divide up the
    reprojection work as well, but some work (such as the PSF correction, if
    applied during this process) is repeated with each chunk, so there is a
    speed--memory tradeoff.

    Parameters
    ----------
    files : ``Iterable`` of ``str``
        A list of file paths, referring to the set of input images
    percentiles : ``float`` or ``Iterable``
        The percentile value or values to use. Since computing multiple
        percentile values is almost free once all the images have been
        reprojected and stacked, this function can accept multiple percentile
        values and return multiple starmaps. This can be very useful when
        comparing different percentile values.
    frame_count : ``bool``, optional
        Whether to track and return the number of input images contributing to
        each pixel in the output image.
    attribution : ``bool``, optional
        If True, this function also returns an attribution array. For each
        pixel in the output skymap, this array contains an index into the list
        of filenames, indicating which file contributed the value selected for
        the output map. (In practice, the output values are interpolated
        between the two input values closest to the exact percentile location,
        and it's the closest of those values that is called the source.)
    processor : ``ImageProcessor``, optional
        An instance of a class providing functions allowing the handling of the
        input images to be customized. This class is responsible for loading
        images from files, pre-processing them before being reprojected, and
        post-processing them after reprojection but before stacking. If not
        provided, a default implementation loads data from FITS files and does
        nothing else. Must be pickleable to support parallel processing.
    reducer : `StackReducer`, optional
        An instance of a class with a ``reduce_strip`` method that reduces the
        stack of images to an output map. See `StackReducer` for more details.
    ra_bounds, dec_bounds : ``Iterable`` of ``float``, optional
        If provided, the bounds to use for the output star map (instead of
        producing a full all-sky map). If not provided, the output map spans
        all of right ascension, but the input images are used to determine
        appropriate declination bounds (ensuring that all input images are
        contained in the bounds).
    stack_all : ``bool``, optional
        For debugging---after the first chunk of the starfield has been
        computed, return the full accumulation array as well as the starfield
        (which is empty in all but the first chunk). This can be very useful to
        inspect the distribution of values at each location. Use in combination
        with ``ra_bounds`` and ``dec_bounds`` to target a particular portion of
        the sky.
    shuffle : ``bool``, optional
        As the input images are reprojected into a given chunk of the output
        skymap, it is likely that many of them won't cover than chunk at all.
        This "no-op" images will result in a very uneven parallel processing
        workload, if the images that do fall within the chunk are clustered
        within the list of input images. To ensure a more even distribution of
        work, the list of input images is randomly shuffled. This can be
        disabled for debugging purposes.

    Returns
    -------
    starfield : `Starfield` or ``List[Starfield]`
        The starfield estimate, including a WCS and, if specified, frame counts
        and attribution information. If multiple maps are produced by
        ``processor``, this will be a list of `Starfield`s.
    """
    # Create the WCS describing the whole-sky starmap
    cdelt = 0.04
    shape = [int(floor(180/cdelt)), int(floor(360/cdelt))]
    starfield_wcs = WCS(naxis=2)
    # n.b. it seems the RA wrap point is chosen so there's 180 degrees included
    # on either side of crpix
    crpix = [shape[1]/2 + .5, shape[0]/2 + .5]
    starfield_wcs.wcs.crpix = crpix
    starfield_wcs.wcs.crval = 180, 0
    starfield_wcs.wcs.cdelt = cdelt, cdelt
    starfield_wcs.wcs.ctype = 'RA---CAR', 'DEC--CAR'
    starfield_wcs.wcs.cunit = 'deg', 'deg'
    
    # Figure out how much of the full sky is covered by our set of images. If
    # we don't go all the way to the celestial poles, we can limit our
    # declination range and save time & memory.
    # Only process every 15th file to speed this up a bit, on the assumption
    # that the on-sky position varies slowly through the image sequence.
    bounds = utils.find_collective_bounds(files[::15], starfield_wcs, key='A')
    
    if ra_bounds is not None:
        # Apply user-specified RA bounds to the output starfield
        (x_min, x_max), _ = starfield_wcs.all_world2pix(ra_bounds, [0, 0], 0)
        x_min = int(x_min)
        x_max = int(x_max)
        starfield_wcs = starfield_wcs[:, x_min:x_max+1]
        x_size = shape[1]
        x_size -= (x_size - x_max)
        x_size -= x_min
        shape[1] = int(x_size)
    # n.b. Since RA is a periodic coordinates, the notion of bounds gets weird
    # without special handling, so don't attempt to automatically clamp the
    # output map in RA.
    
    if dec_bounds is not None:
        # Apply user-specified dec bounds to the output starfield
        _, (y_min, y_max) = starfield_wcs.all_world2pix([10, 10], dec_bounds, 0)
        y_min = int(y_min)
        y_max = int(y_max)
        starfield_wcs = starfield_wcs[y_min:y_max+1, :]
        y_size = shape[0]
        y_size -= (y_size - y_max)
        y_size -= y_min
        shape[0] = int(y_size)
    else:
        # Apply default dec bounds to the output starfield, based on the
        # declination values covered by the input images.
        shape[0] -= shape[0] - bounds[3]
        shape[0] -= bounds[2]
        starfield_wcs = starfield_wcs[bounds[2]:bounds[3]]
    
    # Allocate this later
    starfields = None
    if frame_count:
        count = np.zeros(shape, dtype=int)
    
    # Divide the output starfields into vertical strips, each of which will be
    # processed separately. This avoids extreme memory demands for large sets
    # of input files.
    stride = 5001
    if len(files) > 1000:
        # Auto-reduce the stride when there are lots of input files.
        stride /= len(files) / 1000
        stride = int(stride)
    
    n_chunks = ceil(shape[1] / stride)
    if stack_all:
        n_chunks = 1
    pbar = tqdm(
        total=n_chunks * len(files) + n_chunks * shape[0])
    
    # The order we process these files doesn't matter, and for every section,
    # there will be some input files covering that section and some that don't.
    # Shuffle the file list to get a more even distribution of lots-of-work and
    # no-work files, to benefit the multiprocessing.
    files = [f for f in files]
    fname_to_i = {fname: i for i, fname in enumerate(files)}
    if shuffle:
        random.seed(1)
        random.shuffle(files)
    
    # This is the size of the "working space" array, where we accumulate the
    # values from every image at every pixel in this chunk of the starfield.
    cutout_shape = (len(files), shape[0], stride)
                                                
    with multiprocessing.Pool() as p:
        # Make some memory allocations after the fork
        
        # This is the big honking array that holds a bunch of reprojected
        # images in memory at once. We allocate it only once and keep re-using
        # it, since allocating so much is quite slow.
        starfield_accum = np.empty(cutout_shape)
        
        # Begin looping over output chunks
        for i in range(n_chunks):
            # Work out where we are in the all-sky map
            xstart = stride * i
            xstop = min(shape[1], stride * (i + 1))
            if xstop - xstart < stride:
                # This must be the last iteration
                assert i == n_chunks - 1
                starfield_accum = starfield_accum[:, :, 0:xstop-xstart]
                cutout_shape = starfield_accum.shape
            # imap_unordered only accepts one list of arguments, so bundle up
            # what we need.
            args = zip(
                files,
                repeat(starfield_wcs[:, xstart:xstop]),
                repeat(cutout_shape[1:]),
                repeat(processor))
            n_good = 0
            stack_sources = []
            for (ymin, ymax, xmin, xmax, output, fname) in p.imap_unordered(
                    _process_file, args, chunksize=5):
            # for (ymin, ymax, xmin, xmax, output) in map(process_file_percentile, args):
                pbar.update()
                if output is not None:
                    # In practice, not every input image covers a portion of
                    # each chunk of the output map. As an optimization, instead
                    # of assigning a layer of the accumulation array to each
                    # input image from the start, we assign as we go---each
                    # time a process returns a contribution from an image, we
                    # move to teh next layer of the accumulation array, clear
                    # it, and paste in what we got from the worker process.
                    # This avoids having to clear out the entire array each
                    # time through the loop, and makes it easy to reduce the
                    # work done during the percentile calculation, since we're
                    # not feeding in as many NaNs that have to be filtered.
                    starfield_accum[n_good].fill(np.nan)
                    starfield_accum[n_good, ymin:ymax, xmin:xmax] = output
                    n_good += 1
                    if frame_count:
                        count[:, xstart:xstop][ymin:ymax, xmin:xmax] += (
                            np.isfinite(output))
                    stack_sources.append(fname_to_i[fname])
            
            # Ignore all the slices we didn't use
            starfield_accum_used = starfield_accum[:n_good]
            
            stack_sources = np.array(stack_sources)
            
            # Now that the stacking is complete, we need to calculate the
            # percentile value at each pixel
            
            def args():
                # Generator for arguments as we run the percentile calculation
                # in parallel
                for i in range(starfield_accum_used.shape[1]):
                    # We break up the accumulation array into horizontal
                    # strips, with each strip being one job for the parallel
                    # processing (trying to strike a balance between making
                    # enough work units without making them too small, as it
                    # would be if we did each output pixel as one parallel
                    # job). We copy each chunk to ensure we're not implicitly
                    # sending the whole accumulation array between processes.
                    yield (
                        starfield_accum_used[:, i].copy(),
                        stack_sources if attribution else None,
                        reducer)
            
            for y, res in enumerate(p.imap(
            # for y, res in enumerate(map(
                    _reduce_strip,
                    args(),
                    chunksize=15)):
                    # )):
                pbar.update()
                if attribution:
                    res, srcs = res
                if starfields is None:
                    # Allocate what will be the final output arrays, since we
                    # now know how many output maps to produce
                    starfields = [
                        np.full(shape, np.nan) for _ in range(res.shape[0])]
                    if attribution:
                        attribution_array = np.full(
                            (len(starfields), *shape), -1, dtype=int)
                if attribution:
                    attribution_array[:, y, xstart:xstop] = srcs
                for starfield, r in zip(starfields, res):
                    starfield[y, xstart:xstop] = r
    if attribution:
        mask = np.isnan(starfields[0])
        attribution_array[:, mask] = -1
    pbar.close()
    objects = []
    for i in range(len(starfields)):
        sf = starfields[i]
        if frame_count:
            fc = count
        else:
            fc = None
        if attribution:
            a = attribution_array[i]
        else:
            a = None
        objects.append(Starfield(starfield=sf, wcs=starfield_wcs,
                                 frame_count=fc, attribution=a))
    if len(objects) == 1:
        objects = objects[0]
    if stack_all:
        return starfield_accum_used, objects
    return objects


def _process_file(args):
    """
    Internal function processing a single file. Run in parallel
    """
    fname, starfield_wcs, shape, processor = args
    
    data, wcs = processor.load_image(fname)
    
    # Identify where this image will fall in the whole-sky map
    cdelt = starfield_wcs.wcs.cdelt
    ra_start, dec_start = starfield_wcs.pixel_to_world_values(0, 0)
    ra_stop, dec_stop = starfield_wcs.pixel_to_world_values(
        shape[1] - 1, shape[0] - 1)
    bounds = utils.find_bounds(
        wcs, starfield_wcs, key='A', trim=[70]*4,
        world_coord_bounds=[ra_start - cdelt[0], ra_stop + cdelt[0],
                            dec_start - cdelt[1], dec_stop + cdelt[1]])
    
    if bounds is None:
        # This image doesn't span the portion of the all-sky map now being
        # computed, so we can stop now.
        return [None] * 6
    xmin, xmax, ymin, ymax = bounds

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax >= shape[1]:
        xmax = shape[1]
    if ymax >= shape[0]:
        ymax = shape[0]

    if xmin >= shape[1] or xmax <= 0 or ymin >= shape[0] or ymax <= 0:
        return [None] * 6
    
    data, wcs = processor.preprocess_image(data, wcs, fname)
    
    swcs = starfield_wcs[ymin:ymax, xmin:xmax]
    
    output = reproject.reproject_adaptive(
        (data, wcs), swcs, (ymax-ymin, xmax-xmin),
        return_footprint=False, roundtrip_coords=False,
        boundary_mode='strict',
        conserve_flux=True,
        # This happens to handle the output coordinate wrap-around much better
        center_jacobian=True,
    )
    
    output = processor.postprocess_image(output, swcs, fname)
    
    return ymin, ymax, xmin, xmax, output, fname


def _reduce_strip(args):
    """
    Internal function computing percentiles for a portion of the stack
    """
    data, stack_sources, reducer = args
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
                                message=".*All-NaN slice.*")
        warnings.filterwarnings(action='ignore',
                                message=".*Mean of empty slice*")
        result = reducer.reduce_strip(data)
        if len(result.shape) == 1:
            result = result.reshape((1, -1))
        if stack_sources is not None:
            # We need to figure out which input image contributed the output
            # value for each pixel. Since the exact Nth percentile likely lies
            # between two data points and numpy will interpolate between those
            # points, we search for the closest value and call that the
            # contributor.
            sources = []
            for res in result:
                distances = np.abs(data - res)
                distances = np.nan_to_num(distances, nan=np.inf, posinf=np.inf)
                if np.any(np.isfinite(distances)):
                    i = np.argmin(distances, axis=0)
                    sources.append(stack_sources[i])
                else:
                    sources.append(-1)
            return result, sources
        return result


@dataclass
class Starfield:
    starfield: np.ndarray
    wcs: WCS
    frame_count: np.ndarray | None = None
    attribution: np.ndarray | None = None
    
    def save(self, path: str, overwrite=True):
        """Saves the contents of this object to a file
        
        If a file already exists 

        Parameters
        ----------
        path : ``str`` or file-like object
            The file path at which to save the data
        overwrite : ``bool``
            Whether to overwrite the file if it already exists
        """
        with h5py.File(path, 'w' if overwrite else 'w-') as f:
            f.create_dataset("starfield", data=self.starfield)
            f.create_dataset("wcs", data=self.wcs.to_header_string())
            if self.frame_count is not None:
                f.create_dataset("frame_count", data=self.frame_count)
            if self.attribution is not None:
                f.create_dataset("attribution", data=self.attribution)
    
    @classmethod
    def load(cls, path: str):
        """Loads a `Starfield` that was previously saved to disk

        Parameters
        ----------
        path : ``str`` or file-like object
            The file path to load from

        Returns
        -------
        starfield : `Starfield`
            The object loaded from disk
        """
        with h5py.File(path, 'r') as f:
            # All this [:].copy() syntax ensures we read the data out of the
            # hdf5 file before it's closed
            starfield = f["starfield"][:].copy()
            wcs = WCS(f["wcs"][()])
            frame_count = f.get("frame_count", None)
            if frame_count is not None:
                frame_count = frame_count[:].copy()
            attribution = f.get("attribution", None)
            if attribution is not None:
                attribution = attribution[:].copy()
        return Starfield(starfield=starfield, wcs=wcs, frame_count=frame_count,
                         attribution=attribution)
    
    def plot(self, ax=None, vmin='auto', vmax='auto', pmin=0.1, pmax=99.99,
             grid=False, **kwargs):
        """Plots this starfield
        
        Plots with a gamma correction factor of 1/2.2

        Parameters
        ----------
        ax : ``matplotlib.axes.Axes``, optional
            An axes object on which to plot, or use the current axes if none is
            provided
        vmin : ``float``, optional
            Manually set the colorbar minimum. By default, the starfield's
            0.1th is used (see ``pmin`` below).
        vmax : ``float``, optional
            Manually set the colorbar minimum. By default, the starfield's
            99.99th percentile is used (see ``pmax`` below).
        pmin, pmax : ``float``, optional
            Specify the percentile to be used if vmin/vmax are not provided.
        grid : bool, optional
            Whether to overplot a semi-transparent coordinate grid. Set to a
            float between 0 and 1 to both enable and set the level of
            transparency.
        **kwargs
            Passed to the ``plt.imshow()`` call.

        Returns
        -------
        im
            The return value from the ``plt.imshow`` call
        """
        ax = self._prepare_axes(ax, grid)
        
        if vmin == 'auto':
            vmin = np.nanpercentile(self.starfield, pmin)
        if vmax == 'auto':
            vmax = np.nanpercentile(self.starfield, pmax)
        
        cmap = copy.copy(plt.cm.Greys_r)
        cmap.set_bad('black')
        # Establish plotting defaults, but let kwargs overwrite them
        kwargs = dict(cmap=cmap, origin='lower') | kwargs
        im = ax.imshow(
            self.starfield,
            norm=matplotlib.colors.PowerNorm(
                gamma=1/2.2, vmin=vmin, vmax=vmax),
            **kwargs)
        
        # Set this image to be the one found by plt.colorbar, for instance. But
        # if this manager attribute is empty, pyplot won't accept it.
        if ax.figure.canvas.manager:
            plt.sca(ax)
            plt.sci(im)
        
        return im
    
    def plot_frame_count(self, ax=None, vmin=None, vmax=None, grid=False,
                         **kwargs):
        """Plots this starfield's frame_count array, if present
        
        This array indicates the number of input images that contributed to
        each pixel of the output map.

        Parameters
        ----------
        ax : ``matplotlib.axes.Axes``, optional
            An axes object on which to plot, or use the current axes if none is
            provided
        vmin : ``float``, optional
            Manually set the colorbar minimum.
        vmax : ``float``, optional
            Manually set the colorbar minimum.
        grid : bool, optional
            Whether to overplot a semi-transparent coordinate grid. Set to a
            float between 0 and 1 to both enable and set the level of
            transparency.
        **kwargs
            Passed to the ``plt.imshow()`` call.

        Returns
        -------
        im
            The return value from the ``plt.imshow`` call
        """
        if self.frame_count is None:
            raise ValueError("This Starfield doesn't have a frame_count array")
        
        ax = self._prepare_axes(ax, grid)
        
        # Establish plotting defaults, but let kwargs overwrite them
        kwargs = dict(cmap='viridis', origin='lower') | kwargs
        im = ax.imshow(self.frame_count, vmin=vmin, vmax=vmax, **kwargs)
        
        # Set this image to be the one found by plt.colorbar, for instance. But
        # if this manager attribute is empty, pyplot won't accept it.
        if ax.figure.canvas.manager:
            plt.sca(ax)
            plt.sci(im)
        
        return im
    
    def plot_attribution(self, ax=None, vmin=None, vmax=None, grid=False,
                         mapper=None, **kwargs):
        """Plots this starfield's attribution array, if present
        
        This array indicates the index in the input file list of the file that
        contributed the value at each pixel in the output map.

        Parameters
        ----------
        ax : ``matplotlib.axes.Axes``, optional
            An axes object on which to plot, or use the current axes if none is
            provided
        vmin : ``float``, optional
            Manually set the colorbar minimum.
        vmax : ``float``, optional
            Manually set the colorbar minimum.
        grid : bool, optional
            Whether to overplot a semi-transparent coordinate grid. Set to a
            float between 0 and 1 to both enable and set the level of
            transparency.
        mapper : function
            A function that maps values in the attribution array to other
            values. For example, for PSP/WISPR data, this might be a function
            that converts from the indices in the attribution array to PSP
            encounter numbers. This allows the transformed quantity to be
            easily plotted with world coordinate axis labels. Should be a
            function that accepts as single pixel value as input and returns a
            single modified pixel value.
        **kwargs
            Passed to the ``plt.imshow()`` call.

        Returns
        -------
        im
            The return value from the ``plt.imshow`` call
        """
        if self.attribution is None:
            raise ValueError(
                "This Starfield doesn't have an attribution array")
        
        ax = self._prepare_axes(ax, grid)
        
        if mapper is None:
            image = self.attribution
        else:
            image = np.vectorize(mapper)(self.attribution)
        
        # Establish plotting defaults, but let kwargs overwrite them
        kwargs = dict(cmap='viridis', origin='lower') | kwargs
        im = ax.imshow(image, vmin=vmin, vmax=vmax, **kwargs)
        
        # Set this image to be the one found by plt.colorbar, for instance. But
        # if this manager attribute is empty, pyplot won't accept it.
        if ax.figure.canvas.manager:
            plt.sca(ax)
            plt.sci(im)
        
        return im
    
    def _prepare_axes(self, ax, grid):
        if ax is None:
            ax = plt.gca()
        
        if not isinstance(ax, astropy.visualization.wcsaxes.WCSAxes):
            # We can't apply a WCS projection to existing axes. Instead, we
            # have to destroy and recreate the current axes. We skip that if
            # the axes already are WCSAxes, suggesting that this has been
            # handled already.
            position = ax.get_position().bounds
            ax.remove()
            ax = astropy.visualization.wcsaxes.WCSAxes(
                plt.gcf(), position, wcs=self.wcs)
            plt.gcf().add_axes(ax)
            lon, lat = ax.coords
            lat.set_ticks(np.arange(-90, 90, 15) * u.degree)
            lon.set_ticks(np.arange(-180, 180, 30) * u.degree)
            lat.set_major_formatter('dd')
            lon.set_major_formatter('dd')
            if grid:
                if isinstance(grid, bool):
                    grid = 0.2
                ax.coords.grid(color='white', alpha=grid)
            lon.set_axislabel("Right Ascension")
            lat.set_axislabel("Declination")
        return ax