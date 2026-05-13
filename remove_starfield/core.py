from collections.abc import Iterable
from itertools import repeat
from math import ceil, floor
import multiprocessing
import os
import random

from astropy.wcs import WCS
import numpy as np
import reproject
from tqdm.auto import tqdm
import warnings

from . import ImageProcessor, Starfield, utils, ImageHolder
from .reducers import StackReducer, GaussianReducer


def build_starfield_estimate(
        files: Iterable[str],
        frame_count: bool=False,
        attribution: bool=False,
        processor: ImageProcessor=ImageProcessor(),
        reducer: StackReducer=GaussianReducer(),
        ra_bounds: Iterable[float]=None,
        dec_bounds: Iterable[float]=None,
        starfield_wcs: WCS=None,
        target_mem_usage: float=10,
        map_scale: float=0.04,
        stack_all: bool=False,
        shuffle: bool=True,
        handle_wrap_point: bool=True,
        mask_strategy: str='bounds',
        dtype=float,
        n_procs: int=None) -> Starfield:
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
        the output map. (In practice, the output value is not exactly any of the
        input values---it's instead the input file whose value is closest to the
        output value that is reported.)
    processor : `ImageProcessor`, optional
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
    starfield_WCS : ``WCS``
        Allows a custom WCS to be specified for the starfield WCS, overriding
        the default. If set, ``map_scale``, ``ra_bounds`` and ``dec_bounds``
        will be ignored.
    target_mem_usage : ``float``, optional
        The (approximate) maximum amount of memory to use for the accumulation
        array, as a number of gigabytes. This will determine how many chunks
        the output map is broken into. Higher values will tend to speed up the
        computation. Actual memory usage will likely be less, unless there is a
        portion of the all-sky map spanned by all the input images.
    map_scale : ``float``
        The scale of the output map, in degrees per pixel. (It's the CDELT
        parameter of the output WCS.)
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
    mask_strategy : ``str`` or BlockMasker, optional
        A strategy for reducing the amount of reprojection work. If set to
        "bounds", the edges of each input image are reprojected into the
        starfield frame, and only the bounding box of the result is reprojected
        into. If set tto a BlockMaster instance, that object is used to divide
        the starfield frame into cells, each of which is checked against the
        input image, and only the cells indicated by the BlockMaster are
        reprojected into.
    n_procs : ``int``, optional
        The number of core to use for multi-processing. If unset, the value
        returned by ``os.cpu_count``.

    Returns
    -------
    starfield : `Starfield` or ``List[Starfield]``
        The starfield estimate, including a WCS and, if specified, frame counts
        and attribution information. If multiple maps are produced by
        ``processor``, this will be a list of `Starfield` s.
    stack : ``np.ndarray``
        Returned if ``stack_all==True``. An array of shape ``(n_images x ny x
        chunk_width)`` containing all the samples that contribute to the pixels
        in the first chunk of computation. (Note that ``n_images`` is the
        number of images which contributed to this chunk, not the number of
        input images.)
    stack_sources : ``np.ndarray``
        The index in the input file list of the source file for each position
        along the first axis of ``stack``.
    """
    n_procs = n_procs or os.cpu_count()
    
    if starfield_wcs is None:
        # Create the WCS describing the whole-sky starmap
        shape = [int(floor(180/map_scale)), int(floor(360/map_scale))]
        starfield_wcs = WCS(naxis=2)
        # n.b. it seems the RA wrap point is chosen so there's 180 degrees
        # included on either side of crpix
        crpix = [shape[1]/2 + .5, shape[0]/2 + .5]
        starfield_wcs.wcs.crpix = crpix
        starfield_wcs.wcs.crval = 180, 0
        starfield_wcs.wcs.cdelt = map_scale, map_scale
        starfield_wcs.wcs.ctype = 'RA---CAR', 'DEC--CAR'
        starfield_wcs.wcs.cunit = 'deg', 'deg'
        
        if ra_bounds is not None:
            # Apply user-specified RA bounds to the output starfield
            (x_min, x_max), _ = starfield_wcs.all_world2pix(
                ra_bounds, [0, 0], 0)
            x_min = int(x_min)
            x_max = int(x_max)
            starfield_wcs = starfield_wcs[:, x_min:x_max+1]
            x_size = shape[1]
            x_size -= (x_size - x_max)
            x_size -= x_min
            shape[1] = int(x_size)
        # n.b. Since RA is a periodic coordinates, the notion of bounds gets
        # weird without special handling, so don't attempt to automatically
        # clamp the output map in RA.
        
        if dec_bounds is not None:
            # Apply user-specified dec bounds to the output starfield
            _, (y_min, y_max) = starfield_wcs.all_world2pix(
                [10, 10], dec_bounds, 0)
            y_min = int(y_min)
            y_max = int(y_max)
            starfield_wcs = starfield_wcs[y_min:y_max+1, :]
            y_size = shape[0]
            y_size -= (y_size - y_max)
            y_size -= y_min
            shape[0] = int(y_size)
        else:
            # Figure out how much of the full sky is covered by our set of
            # images. If we don't go all the way to the celestial poles,
            # we can limit our declination range and save time & memory.
            # Only process every 15th file to speed this up a bit, on the
            # assumption that the on-sky position varies slowly through the
            # image sequence.
            bounds = utils.find_collective_bounds(
                files[::15], starfield_wcs, processor=processor)
            # Apply default dec bounds to the output starfield, based on the
            # declination values covered by the input images.
            shape[0] -= shape[0] - bounds[3]
            shape[0] -= bounds[2]
            starfield_wcs = starfield_wcs[bounds[2]:bounds[3]]
        starfield_wcs.array_shape = shape
    else:
        shape = starfield_wcs.array_shape
    
    # Allocate this later
    starfields = None
    if frame_count:
        count = np.zeros(shape, dtype=int)
    
    # Divide the output starfields into vertical strips, each of which will be
    # processed separately. This avoids extreme memory demands for large sets
    # of input files.
    size_of_pixel = np.empty(1).dtype.itemsize
    size_of_column = size_of_pixel * shape[0] * len(files)
    stride = int(target_mem_usage * 1024**3 // size_of_column)
    if stride > shape[1]:
        stride = shape[1]
    
    n_chunks = ceil(shape[1] / stride)
    if stack_all:
        n_chunks = 1
    pbar_stack = tqdm(total=n_chunks * len(files), desc="Reprojecting")
    pbar_reduce = tqdm(total=n_chunks * shape[0], desc="Reducing")
    
    # The order we process these files doesn't matter, and for every section,
    # there will be some input files covering that section and some that don't.
    # Shuffle the file list to get a more even distribution of lots-of-work and
    # no-work files, to benefit the multiprocessing.
    fname_to_i = {fname: i for i, fname in enumerate(files)}
    if shuffle:
        random.seed(1)
        random.shuffle(files)
    
    # This is the size of the "working space" array, where we accumulate the
    # values from every image at every pixel in this chunk of the starfield.
    cutout_shape = (len(files), shape[0], stride)
    
    with multiprocessing.Pool(processes=n_procs) as p:
        # Make some memory allocations after the fork
        
        # This is the big honking array that holds a bunch of reprojected
        # images in memory at once. We allocate it only once and keep re-using
        # it, since allocating so much is quite slow.
        starfield_accum = np.empty(cutout_shape, dtype=dtype)
        
        # Begin looping over output chunks
        for i in range(n_chunks):
            # Work out where we are in the all-sky map
            xstart = stride * i
            xstop = min(shape[1], stride * (i + 1))
            if xstop - xstart < stride:
                # This must be the last iteration
                assert i == n_chunks - 1
                starfield_accum = starfield_accum[:, :, 0:xstop-xstart]
            # imap_unordered only accepts one list of arguments, so bundle up
            # what we need.
            args = zip(
                files,
                repeat(starfield_wcs[:, xstart:xstop]),
                repeat(processor),
                repeat(handle_wrap_point),
                repeat(dtype),
                repeat(mask_strategy))
            n_good = 0
            stack_sources = []
            reproject_chunk_size = min(5, int(len(files) / n_procs / 3))
            reproject_chunk_size = max(reproject_chunk_size, 1)

            worker_fcn = _process_file if mask_strategy == 'bounds' else _process_file_block_mask
            if n_procs == 1:
                # Don't go parallel if we don't have to---makes it easier to use a debugger
                iterator = map(worker_fcn, args)
            else:
                iterator = p.imap_unordered(
                   worker_fcn, args, chunksize=reproject_chunk_size)
            for (reprojected_results, fname) in iterator:
                pbar_stack.update()
                if reprojected_results is None:
                    continue
                # In practice, not every input image covers a portion of each
                # chunk of the output map. As an optimization, instead of
                # assigning a layer of the accumulation array to each input
                # image from the start, we assign as we go---each time a
                # process returns a contribution from an image, we move to
                # the next layer of the accumulation array, clear it,
                # and paste in what we got from the worker process. This
                # avoids having to clear out the entire array each time
                # through the loop, and makes it easy to reduce the work done
                # during the percentile calculation, since we're not feeding
                # in as many NaNs that have to be filtered.
                starfield_accum[n_good].fill(np.nan)
                for slice, output in reprojected_results:
                    starfield_accum[n_good, *slice] = output
                    if frame_count:
                        count[:, xstart:xstop][slice] += (
                            np.isfinite(output))
                n_good += 1
                stack_sources.append(fname_to_i[fname])
            pbar_stack.refresh()
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

            if n_procs == 1:
                iterator = map(_reduce_strip, args())
            else:
                iterator = p.imap(_reduce_strip, args(), chunksize=20)
            for y, res in enumerate(iterator):
                pbar_reduce.update()
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
            pbar_reduce.refresh()
    if attribution:
        mask = np.isnan(starfields[0])
        attribution_array[:, mask] = -1
    pbar_stack.close()
    pbar_reduce.close()
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
        return objects, starfield_accum_used, stack_sources
    return objects


def _process_file(args):
    """
    Internal function processing a single file. Run in parallel
    """
    fname, starfield_wcs, processor, handle_wrap_point, dtype, _ = args

    shape = starfield_wcs.array_shape

    image_holder = processor.load_image(fname)

    # Identify where this image will fall in the whole-sky map
    edges_x, edges_y = utils.points_along_edge(shape, trim=[.1, .1, .1, .1],
                                               n_pts=-1, separate_edges=True)
    ras, decs = starfield_wcs.pixel_to_world_values(edges_x[3], edges_y[3])
    ra_start = np.min(ras)

    ras, decs = starfield_wcs.pixel_to_world_values(edges_x[1], edges_y[1])
    ra_stop = np.max(ras)
    ra_stop = utils.wrap_inside_period(ra_stop, ra_start, 360)

    ras, decs = starfield_wcs.pixel_to_world_values(edges_x[0], edges_y[0])
    dec_start = np.min(decs)

    ras, decs = starfield_wcs.pixel_to_world_values(edges_x[2], edges_y[2])
    dec_stop = np.max(decs)

    cdelt = starfield_wcs.wcs.cdelt
    bounds_sets = utils.find_bounds(
        image_holder.wcs, starfield_wcs, processor=processor,
        world_coord_bounds=[ra_start - cdelt[0], ra_stop + cdelt[0],
                            dec_start - cdelt[1], dec_stop + cdelt[1]],
        ra_wrap_point=ra_start, wrap_aware=True)

    if bounds_sets is None:
        # This image doesn't span the portion of the all-sky map now being
        # computed, so we can stop now.
        return None, None

    reprojected_results = []
    for bounds in bounds_sets:
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
            continue

        image_holder = processor.preprocess_image(image_holder)

        s = np.s_[ymin:ymax, xmin:xmax]
        swcs = starfield_wcs[s]

        output = np.empty((ymax - ymin, xmax - xmin), dtype=dtype)
        reproject.reproject_adaptive(
            (image_holder.data, image_holder.wcs), swcs,
            (ymax - ymin, xmax - xmin),
            output_array=output,
            return_footprint=False, roundtrip_coords=False,
            boundary_mode='strict',
            conserve_flux=True,
            # This seems to handle the output coordinate wrap-around much better
            center_jacobian=handle_wrap_point,
        )

        output = processor.postprocess_image(output, swcs, image_holder)
        reprojected_results.append((s, output))
    
    if len(reprojected_results):
        return reprojected_results, fname
    return None, None


def _process_file_block_mask(args):
    """
    Internal function processing a single file. Run in parallel
    """
    fname, starfield_wcs, processor, handle_wrap_point, dtype, masker = args

    shape = starfield_wcs.array_shape

    image_holder = processor.load_image(fname)
    image_holder = processor.preprocess_image(image_holder)

    image_strips = masker.identify_strips(shape, starfield_wcs, image_holder)

    reprojected_results = []
    for strip in image_strips:
        swcs = starfield_wcs[strip]

        output = np.empty(swcs.array_shape, dtype=dtype)
        reproject.reproject_adaptive(
            (image_holder.data, image_holder.wcs), swcs,
            swcs.array_shape,
            output_array=output,
            return_footprint=False, roundtrip_coords=False,
            boundary_mode='strict',
            conserve_flux=True,
        )

        output = processor.postprocess_image(output, swcs, image_holder)
        reprojected_results.append((strip, output))

    if len(reprojected_results):
        return reprojected_results, fname
    return None, None


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


class BlockMasker:
    """Class to identify strips of the output plane to reproject into.
    
    For input images that don't fill the whole frame, we can save a good chunk of time by not reprojecting the empty
    parts of the input images. This class handles that. The output frame is divided in to a grid of cells. For each
    input image, the center of each cell is transformed into the input frame, and a cell around that center is
    sampled to determine if it's empty. Output cells that map to non-empty input cells are joined into contiguous
    strips, each each strip of contiguous non-empty cells is reprojected independently. The strips don't include
    empty cells, so we don't spend time reprojecting the empty data, and the use of cells and strips of joined cells
    means we spend very little time determining which areas to reproject, and our reproject calls are still big
    enough that we don't add much overhead from multiple calls.

    One limitation is that only the location of the cells in the output frame is transformed to the input images,
    not the shape or size of the cell. This means that at the edges of the empty region, our marking of cells as full
    or empty is approximate, and the edges of the good region of input images might still be ignored. This is
    probably a price worth paying.

    This class is designed with pieces that can be overridded in a subclass, so custom logic can be implmented.
    """
    def __init__(self, x_wsize: int = 128, y_wsize: int = 128):
        """

        Parameters
        ----------
        x_wsize, y_size : int
            The size of the cells in the output frame (i.e. the starfield map). Ideally this will evenly divide the starfield
            size.
        """
        self.x_wsize, self.y_wsize = x_wsize, y_wsize

    def generate_blocks(self, starmap_shape: tuple):
        """
        Produce a list of cells in the output frame.

        Parameters
        ----------
        starmap_shape : tuple
            the shape of the output frame

        Returns
        -------
        blocks : np.ndarray
            A (2, n_blocks_y, n_blocks_x) array, containing in the first dimension the x and y coordinate at which each block starts.
        block_centers : np.ndarray
            A (2, n_blocks_y, n_blocks_x) array, containing in the first dimension the central x and y coordinate of each block.
        """
        xx = np.arange(0, starmap_shape[1], self.x_wsize)
        yy = np.arange(0, starmap_shape[0], self.y_wsize)
        blocks = np.array(np.meshgrid(xx, yy))
        centers = blocks.copy()
        centers[0] += self.x_wsize // 2
        centers[1] += self.y_wsize // 2
        return blocks, centers

    def check_blocks(self, block_centers: np.ndarray, starmap_wcs: WCS, image_holder: ImageHolder):
        """
        Produces a 2D mask indicating whether the corresponding blocks are filled.

        Parameters
        ----------
        block_centers : `np.ndarray`
            The block centers returned by generate_blocks
        starmap_wcs : WCS
            The starmap `WCS`
        image_holder : `ImageHolder`
            The input image

        Returns
        -------
        should_use_block : `np.ndarray`
            A binary mask with shape (n_blocks_y, n_blocks_x) indicating whether each block should be used.
        """
        image_x, image_y = image_holder.wcs.world_to_pixel(
            starmap_wcs.pixel_to_world(block_centers[0], block_centers[1]))
        should_use_block = np.empty(block_centers[0].shape, dtype=bool)
        for i in range(block_centers[0].shape[0]):
            for j in range(block_centers[0].shape[1]):
                should_use_block[i, j] = self.check_block((image_x[i, j], image_y[i, j]), image_holder)
        return should_use_block

    def check_block(self, block_center: tuple, image_holder: ImageHolder):
        """
        Takes a single block in the output frame and determines whether it should be reprojected into.

        If custom logic is required for a data set, this is likely the method to override.

        Parameters
        ----------
        block_center : tuple
            The coordinates in the input image that correspond to the center of the block in the output frame
        image_holder : ImageHolder
            The input image

        Returns
        -------
        block_is_good: bool
        """
        x, y = block_center
        x, y = int(x), int(y)
        if not (0 < x < image_holder.data.shape[1]):
            return False
        if not (0 < y < image_holder.data.shape[0]):
            return False
        window = image_holder.data[
            max(0, y - self.y_wsize // 2): min(y + self.y_wsize // 2, image_holder.data.shape[1]),
            max(0, x - self.x_wsize // 2): min(x + self.x_wsize // 2, image_holder.data.shape[0])]
        return np.any((window != 0) * np.isfinite(window))

    def identify_strips(self, starmap_shape: tuple, starmap_wcs: WCS, image_holder: ImageHolder):
        """
        Produce a list of strips to reproject into.

        This is the entry point for remove_starfield. A grid of blocks is generated and checked to determine which
        blocks should be reprojected into. Contiguous "good" blocks are merged into strips, and those strips are
        returned as slice objects.

        Parameters
        ----------
        starmap_shape : tuple
            The starmap shape
        starmap_wcs : WCS
            The starmap WCS
        image_holder : ImageHolder
            The input image to check.

        Returns
        -------
        strips : list[slice]
            The strips to reproject into
        """
        strips = []
        blocks, block_centers = self.generate_blocks(starmap_shape)
        block_is_good = self.check_blocks(block_centers, starmap_wcs, image_holder)
        for i in range(blocks.shape[1]):
            x_start = 0
            for j in range(blocks.shape[2]):
                x, y = blocks[:, i, j]
                if not block_is_good[i, j]:
                    if x != x_start:
                        s = np.s_[y:y + self.y_wsize, x_start:x]
                        strips.append(s)
                    x_start = x + self.x_wsize
                elif j == blocks.shape[2] - 1:
                    s = np.s_[y:y + self.y_wsize, x_start:x + self.x_wsize]
                    strips.append(s)

        return strips