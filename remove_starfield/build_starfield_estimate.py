from collections.abc import Iterable
from itertools import repeat
from math import ceil, floor
import multiprocessing
import random

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import reproject
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


def build_starfield_estimate_percentile(
        files: Iterable[str],
        percentiles: float | Iterable[float],
        frame_count: bool=False,
        attribution: bool=False,
        processor: ImageProcessor=ImageProcessor(),
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
        A class providing functions allowing the handling of the input images
        to be customized. This class is responsible for loading images from
        files, pre-processing them before being reprojected, and
        post-processing them after reprojection but before stacking. If not
        provided, a default implementation loads data from FITS files and does
        nothing else. Must be pickleable to support parallel processing.
    ra_bounds, dec_bounds : ``Iterable`` of ``float``, optional
        If provided, the bounds to use for the output star map (instead of
        producing a full all-sky map). If not provided, the output map spans
        all of right ascension, but the input images are used to determine
        appropriate declination bounds (ensuring that all input images are
        contained in the bounds).
    stack_all : ``bool``, optional
        For debugging---after all images have been stacked in the first chunk,
        simply return the accumulation array.
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
    starfield : ``np.ndarray`` or ``List[np.ndarray]``
        The starfield estimate, or a list of estimates if multiple percentile
        values were provided.
    starfield_wcs : ``astropy.wcs.WCS``
        A WCS object describing the starfield array
    frame_count : ``np.ndarray``
        Provided if ``frame_count==True``. An array indicating the number of
        input images that contributed to each pixel in the starfield estimate.
    sources : ``np.ndarray`` or ``List[np.ndarray]``
        Provided if ``attribution==True``. An array indicating the source file
        for each value represented in the starfield estimate, as indexes into
        the list of filenames. If multiple percentile values were provided,
        multiple source arrays will be produced.
    """
    percentiles_orig = percentiles
    percentiles = np.atleast_1d(np.asarray(percentiles))
    
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
    
    # Allocate what will be the final output arrays
    starfields = [np.full(shape, np.inf) for p in percentiles]
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
        starfield_accum = np.empty(cutout_shape, dtype=starfields[0].dtype)

        if attribution:
            attribution_array = np.full(
                (len(percentiles), *shape), -1, dtype=int)
        
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
            
            if stack_all:
                if frame_count:
                    return starfield_accum_used, stack_sources, count
                return starfield_accum_used, stack_sources
            
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
                        percentiles,
                        stack_sources if attribution else None)
            
            for y, res in enumerate(p.imap(
            # for y, res in enumerate(map(
                    _find_percentile_for_strip,
                    args(),
                    chunksize=15)):
                    # )):
                pbar.update()
                if attribution:
                    res, srcs = res
                    attribution_array[:, y, xstart:xstop] = srcs
                for starfield, r in zip(starfields, res):
                    starfield[y, xstart:xstop] = r
    if attribution:
        mask = np.isnan(starfields[0])
        attribution_array[:, mask] = -1
    pbar.close()
    if not isinstance(percentiles_orig, Iterable):
        starfields = starfields[0]
    retval = starfields, starfield_wcs
    if frame_count:
        retval += (count,)
    if attribution:
        retval += (attribution_array,)
    return retval


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


def _find_percentile_for_strip(args):
    """
    Internal function computing percentiles for a portion of the stack
    """
    data, percentiles, stack_sources = args
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',
                                message=".*All-NaN slice.*")
        warnings.filterwarnings(action='ignore',
                                message=".*Mean of empty slice*")
        result = np.nanpercentile(data, percentiles, axis=0)
        if stack_sources is not None:
            # We need to figure out which input image contributed the output
            # value for each pixel. Since the exact Nth percentile likely lies
            # between two data points and numpy will interpolate between those
            # points, we search for the closest value and call that the
            # contributor.
            sources = []
            for pctl in result:
                distances = np.abs(data - pctl)
                distances = np.nan_to_num(distances, nan=np.inf, posinf=np.inf)
                if distances.size:
                    i = np.argmin(distances, axis=0)
                    sources.append(stack_sources[i])
                else:
                    sources.append('')
            return result, sources
        return result



