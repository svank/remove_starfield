from collections.abc import Iterable
import os

import astropy.units as u
import astropy.visualization.wcsaxes
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np

from .processor import ImageProcessor

def find_collective_bounds(wcses, wcs_target, trim=(0, 0, 0, 0),
                           processor: ImageProcessor=None):
    """
    Finds the bounding coordinates for a set of input images.
    
    Calls `find_bounds` for each provided header, and finds the bounding box in
    the output coordinate system that will contain all of the input images.
    
    Parameters
    ----------
    wcses : Iterable
        Either a list of WCSes, or a list of lists of WCSes. If the latter,
        ``trim`` can be a list of trim values, one for each of the lists of
        Headers. Instead of WCSes, each instance can be the path to a FITS
        file.
    wcs_target : ``astropy.wcs.WCS``
        A WCS object describing an output coordinate system.
    trim : ``tuple`` or ``list``
        How many rows/columns to ignore from the input image. In order,
        (left, right, bottom, top). If ``wcses`` is a list of lists of Headers,
        this can be (but does not have to be) be a list of tuples of trim
        values, one for each list of Headers.
    processor : `ImageProcessor`
        An `ImageProcessor` to load FITS files. Only required if file paths are
        passed in for ``wcses``.
    
    Returns
    -------
    bounds : tuple
        The bounding coordinates. In order, (left, right, bottom, top).
    """
    
    if isinstance(wcses, (WCS, str)):
        wcses = [[wcses]]
    if isinstance(wcses[0], (WCS, str)):
        wcses = [wcses]
    if not isinstance(trim[0], Iterable):
        trim = [trim] * len(wcses)
    
    bounds = []
    for w, t in zip(wcses, trim):
        bounds += [find_bounds(wcs, wcs_target, trim=t, processor=processor)
                   for wcs in w]
    bounds = np.array(bounds).T
    return (np.min(bounds[0]), np.max(bounds[1]),
            np.min(bounds[2]), np.max(bounds[3]))


def find_bounds(wcs, wcs_target, trim=(0, 0, 0, 0),
                processor: ImageProcessor=None, world_coord_bounds=None,
                ra_wrap_point=0, wrap_aware=False):
    """Finds the pixel bounds of a FITS header in an output WCS.
    
    The edges of the input image are transformed to the coordinate system of
    ``wcs_target``, and the extrema of these transformed coordinates are found.
    In other words, this finds the size of the output image that is required to
    bound the reprojected input image.
    
    Optionally, handles the case that the x axis of the output WCS is periodic
    and the input WCS straddles the wrap point. Two sets of bounds are
    returned, for the two halves of the input WCS.
    
    Parameters
    ----------
    wcs : ``str`` or ``WCS``
        A WCS describing an input image's size and coordinate system,
        or the path to a FITS file whose header will be loaded.
    wcs_target : ``astropy.wcs.WCS``
        A WCS object describing an output coordinate system.
    trim : ``tuple``
        How many rows/columns to ignore from the input image. In order,
        ``(left, right, bottom, top)``.
    processor : `ImageProcessor`
        An `ImageProcessor` to load FITS files. Only required if a file path is
        passed in for ``wcs``.
    world_coord_bounds : ``list``
        Edge pixels of the image that fall outside these world coordinates are
        ignored. Must be a list of four values ``[RAmin, RAmax, Decmin, Decmax]``.
        Any value can be ``None`` to not provide a bound. The RA bounds are
        used for find the y bounds, and the dec bounds are separately used for
        finding the x bounds.
    ra_wrap_point : ``float``
        Defines the "start point" for right ascension. RA is a periodic
        quantity, and for the purposes of applying ``world_coord_bounds``, RA
        values will be wrapped to fall within the range
        ``(ra_wrap_point, ra_wrap_point + 360)``.
    wrap_aware : boolean
        Whether to heuristically check for and handle the case that the image
        straddles the wrap point of the periodic x axis.
    
    Returns
    -------
    bounds : list of tuples
        The bounding coordinates. In order, (left, right, bottom, top). One or
        two such tuples are returned, depending on whether the input WCS
        straddles the output's wrap point.
    """
    # Parse inputs
    if isinstance(wcs, str):
        ih = processor.load_image(wcs)
        wcs = ih.wcs
    
    xs, ys = points_along_edge(wcs.array_shape, trim, n_pts=100)
    
    ra, dec = wcs.pixel_to_world_values(xs, ys)
    assert not np.any(np.isnan(ra)) and not np.any(np.isnan(dec))
    
    if world_coord_bounds is not None:
        assert len(world_coord_bounds) == 4
        if world_coord_bounds[0] is None:
            world_coord_bounds[0] = -np.inf
        if world_coord_bounds[2] is None:
            world_coord_bounds[2] = -np.inf
        if world_coord_bounds[1] is None:
            world_coord_bounds[1] = np.inf
        if world_coord_bounds[3] is None:
            world_coord_bounds[3] = np.inf
        ra_bounds = world_coord_bounds[0:2]
        dec_bounds = world_coord_bounds[2:4]
        ra = wrap_inside_period(ra, ra_wrap_point, 360)
        f_for_x = (dec_bounds[0] <= dec) * (dec <= dec_bounds[1])
        f_for_y = (ra_bounds[0] <= ra) * (ra <= ra_bounds[1])
        if not np.any(f_for_x) or not np.any(f_for_y):
            return None
    else:
        f_for_x = np.ones(len(ra), dtype=bool)
        f_for_y = f_for_x
    
    cx_orig, cy = wcs_target.world_to_pixel_values(ra, dec)
    cx = cx_orig[f_for_x]
    cy = cy[f_for_y]
    
    if not wrap_aware:
        return (int(np.floor(np.min(cx))),
                int(np.ceil(np.max(cx))),
                int(np.floor(np.min(cy))),
                int(np.ceil(np.max(cy))))
    
    ranges = []
    
    cx_hist, cx_hist_edges = np.histogram(cx, bins=11)
    if cx_hist[0] > 0 and cx_hist[-1] > 0 and cx_hist[len(cx_hist) // 2] == 0:
        divider = cx_hist_edges[len(cx_hist) // 2]
        for sign in (cx_orig - divider, divider - cx_orig):
            filter = sign > 0
            cxf = cx[filter[f_for_x]]
            cyf = cy[filter[f_for_y]]
            if len(cxf) and len(cyf):
                xmin = cxf.min()
                xmax = cxf.max()
                ymin = cyf.min()
                ymax = cyf.max()
                ranges.append((xmin, xmax, ymin, ymax))
    else:
        ranges.append((cx.min(), cx.max(), cy.min(), cy.max()))
    for i, r in enumerate(ranges):
        r = (
            int(np.floor(r[0])),
            int(np.ceil(r[1])),
            int(np.floor(r[2])),
            int(np.ceil(r[3])))
        ranges[i] = r
    if not len(ranges):
        return None
    return ranges


def wrap_inside_period(values, period_start, period_size):
    return (values - period_start) % period_size + period_start


def points_along_edge(shape, trim=(0, 0, 0, 0), n_pts=-1, separate_edges=False):
    # Generate pixel coordinates along the edges, accounting for the trim
    # values
    left = -.5 + trim[0]
    right = shape[1] - .5 - trim[1]
    bottom = -.5 + trim[2]
    top = shape[0] - .5 - trim[3]
    xs = np.linspace(left, right, shape[1] if n_pts == -1 else n_pts)
    ys = np.linspace(bottom, top, shape[0] if n_pts == -1 else n_pts)
    edgex = [xs, # bottom edge
             np.full(len(ys), xs[-1]), # right edge
             xs, # top edge
             np.full(len(ys), xs[0])] # left edge
    edgey = [np.full(len(xs),ys[0]), # bottom edge
             ys, # right edge
             np.full(len(xs), ys[-1]), # top edge
             ys] # left edge
    if not separate_edges:
        edgex = np.concatenate(edgex)
        edgey = np.concatenate(edgey)
    return edgex, edgey


def prepare_axes(ax, wcs=None, grid=False):
    """Applies a WCS projection to an Axes and sets up axis labels"""
    if ax is None:
        ax = plt.gca()
    
    if wcs is None:
        return ax
    
    if not isinstance(ax, astropy.visualization.wcsaxes.WCSAxes):
        # We can't apply a WCS projection to existing axes. Instead, we
        # have to destroy and recreate the current axes. We skip that if
        # the axes already are WCSAxes, suggesting that this has been
        # handled already.
        position = ax.get_position().bounds
        ax.remove()
        ax = astropy.visualization.wcsaxes.WCSAxes(
            plt.gcf(), position, wcs=wcs)
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


def test_data_path(*segments):
    """Returns the path to the test data directory, with segments appended"""
    return os.path.join(os.path.dirname(__file__),
                        'tests', 'test_data', *segments)


def data_path(*segments):
    """
    Returns the path to the package data directory, with segments appended
    """
    return os.path.join(os.path.dirname(__file__), 'data', *segments)


def find_data_and_celestial_wcs(hdul, data=True, wcs=True, header=False):
    # If the FITS file is compressed, the first HDU has no data. Search for the
    # first non-empty hdu
    hdu = 0
    while hdul[hdu].data is None:
        hdu += 1

    if wcs:
        hdr = hdul[hdu].header
        # Search for a celestial WCS
        for key in ' ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            found_wcs = WCS(hdr, hdul, key=key)
            ctypes = sorted(found_wcs.wcs.ctype)
            ctypes = [c[:3] for c in ctypes]
            if 'DEC' in ctypes and 'RA-' in ctypes:
                break
        else:
            raise ValueError("No celestial WCS found")
    
    ret = []
    if data:
        ret.append(hdul[hdu].data)
    if wcs:
        ret.append(found_wcs)
    if header:
        ret.append(hdul[hdu].header)
    if len(ret) == 1:
        return ret[0]
    return ret
