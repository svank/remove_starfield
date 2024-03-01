from itertools import chain
from math import ceil, floor

from astropy.io import fits
from astropy.wcs import WCS
import ipywidgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import utils


def extract_flux(img_sequence, x, y, aperture_r=5, gap=2, annulus_thickness=3,
                 skip_edge_stars=False):
    """Measures a flux aperture-photometry style
    
    At the x/y coordinate pair given, the pixel values are summed through a
    circular region centered on the (rounded) x/y coordinate with radius
    ``aperture_r`` pixels. Then, an annulus is drawn with inner radius
    ``aperture_r + gap`` and outer radius ``aperture_r + gap +
    annulus_thickness``. Within this annulus, the median pixel value is
    selected as the estimated background level. That value is scaled for the
    size of the central region and then subtracted from the summed value.

    Parameters
    ----------
    img_sequence : ``np.ndarray`` or ``Iterable[np.ndarray]``
        An image or a sequence of images from which to extract fluxes
    x, y : ``float``
        The location at which to center the aperture. Will be rounded to an
        integer.
    aperture_r : ``int``
        The radius of the aperture, in pixels
    gap : ``int``
        The distance between the aperture edge and the annulus edge, in pixels
    annulus_thickness : ``int``
        The thickness of the annulus, in pixels
    skip_edge_stars : ``bool``
        If True, if the annulus crosses the edge of the image, size-zero array
        of fluxes will be returned. If False and this occurs, an exception is
        raised.

    Returns
    -------
    fluxes : ``np.ndarray``
        An array of flux values, one for each image in ``img_sequence``.
    """
    if type(img_sequence) == np.ndarray and img_sequence.ndim == 2:
        img_sequence = [img_sequence]
    
    cutout_width = aperture_r + gap + annulus_thickness + 1
    fluxes = []
    
    for img in img_sequence:
        x, y = int(np.round(x)), int(np.round(y))
        if not (cutout_width < y <= img.shape[0] - cutout_width - 1
                and cutout_width < x <= img.shape[1] - cutout_width - 1):
            if skip_edge_stars:
                continue
            raise ValueError("Cutout does not fit in image")
        cutout = img[y - cutout_width : y + cutout_width + 1,
                     x - cutout_width : x + cutout_width + 1]
        x0 = cutout_width
        y0 = cutout_width
        xs = np.arange(cutout.shape[1]) - x0
        ys = np.arange(cutout.shape[0]) - y0
        xs, ys = np.meshgrid(xs, ys)
        r = np.sqrt(xs**2 + ys**2)
        
        bg = cutout[(r > aperture_r + gap)
                    * (r <= aperture_r + gap + annulus_thickness)]
        bg_value = np.nanmedian(bg[np.isfinite(bg)])
        
        center = (r <= aperture_r)
        central_flux = cutout[center]
        central_flux = central_flux[np.isfinite(central_flux)]
        central_flux = np.sum(central_flux)
        bg_value *= np.sum(center)
        
        fluxes.append(central_flux - bg_value)
    
    fluxes = np.array(fluxes)
    return fluxes


def illustrate_flux(img_sequence, aperture_r=2, gap=1, annulus_thickness=2):
    """Utility to aid in setting photometry parameters
    
    Creates an interactive Jupyter interface, allowing the user to select an
    image and then a star within that image. Shows a close-up view of the image
    with an aperture and annulus overdrawn, with sliders to vary the aperture
    and annulus sizes.
    
    Uses the in-package star catalog to automatically find the locations of
    reasonably-bright stars.

    Parameters
    ----------
    img_sequence : ``Tuple[np.ndarray, WCS]`` or list thereof
        The image to view, or a sequence of images to view. Must provide a
        tuple of image data and WCS for each image.
    aperture_r : ``int``, optional
        Initial value to use for the aperture
    gap : ``int``, optional
        Initial value to use for the aperture--annulus gap
    annulus_thickness : ``int``, optional
        Initial value to use for the annulus
    """
    if type(img_sequence[0]) == np.ndarray:
        img_sequence = [img_sequence]
    
    stars_in_images = [find_expected_stars_in_frame(
                           wcs, dim_cutoff=5, bright_cutoff=-10)
                       for img, wcs in img_sequence]
    
    def f(i, star_n, aperture_r=aperture_r, gap=gap,
            annulus=annulus_thickness):
        (stars_x, stars_y, stars_vmag, stars_ra, stars_dec,
            all_stars_x, all_stars_y) = stars_in_images[i]
        img, wcs = img_sequence[i]
        x, y = stars_x[star_n], stars_y[star_n]
        x, y = np.round([x, y])
        flux = extract_flux(img, x, y, aperture_r, gap, annulus,
                            skip_edge_stars=True)
        vmin, vmax = np.nanpercentile(img, [1, 99])
        plt.imshow(img, origin='lower', norm=matplotlib.colors.PowerNorm(
                gamma=1/2.2, vmin=vmin, vmax=vmax), cmap='Greys_r')
        plt.ylim(y-12, y+12)
        plt.xlim(x-12, x+12)
        plt.scatter([x], [y])
        
        thetas = np.linspace(0, 2*np.pi)
        r = aperture_r
        plt.plot(x + r * np.cos(thetas), y + r * np.sin(thetas), c='C1')
        r = aperture_r + gap
        plt.plot(x + r * np.cos(thetas), y + r * np.sin(thetas), c='C1')
        r = aperture_r + gap + annulus
        plt.plot(x + r * np.cos(thetas), y + r * np.sin(thetas), c='C1')
        if len(flux):
            plt.title(f"Flux: {flux[0]}")
        else:
            plt.title("Sample region falls outside image")
    
    star_n_widget = ipywidgets.IntSlider(min=0, max=1,
                                         description="Select a star")
    if len(img_sequence) > 1:
        img_i_widget = ipywidgets.IntSlider(min=0, max=len(img_sequence)-1,
                                            description="Select image")
    else:
        img_i_widget = ipywidgets.fixed(0)
    
    def update_n_range(*args):
        star_n_widget.max = len(stars_in_images[img_i_widget.value][0]) - 1
    update_n_range()
    star_n_widget.observe(update_n_range, 'value')
    
    return ipywidgets.interact(
            f,
            i=img_i_widget,
            star_n=star_n_widget,
            aperture_r=(1, 15),
            gap=(0, 15),
            annulus=(1, 15),
    )


def find_expected_stars_in_frame(input, dim_cutoff=8, bright_cutoff=2,
                                 trim=(0,0,0,0)):
    """Consults a stellar catalog to find the stars that should be in an image

    Parameters
    ----------
    input : ``str`` or ``WCS``
        A path to a FITS file to load, or a WCS object
    dim_cutoff : ``float``
        The maximum magnitude of stars to consider
    bright_cutoff : ``float``
        The minimum magnitude of stars to consider
    trim : ``int`` or ``Tuple[int]``
        Exclude stars within a certain distance of the edge of the image. If an
        ``int``, stars within that many pixels of any edge are excluded. If a
        sequence of ``int``, the exclusion zone for each edge can be set. The
        order of the tuple is ``(left, right, bottom, top)``.

    Returns
    -------
    stars_x : ``np.ndarray``
        The x coordinate of each star
    stars_y : ``np.ndarray``
        The x coordinate of each star
    stars_vmag : ``np.ndarray``
        The magnitude of each star
    stars_ra : ``np.ndarray``
        The right ascension coordinate of each star
    stars_dec : ``np.ndarray``
        The declination coordinate of each star
    """
    if isinstance(trim, int):
        trim = [trim] * 4
    
    if isinstance(input, str):
        with fits.open(input) as hdul:
            wcs = utils.find_data_and_celestial_wcs(hdul, data=False)
    else:
        wcs = input
    
    # Generate points all along the edge of the image so we can work out its
    # bounds
    left = 0 + trim[0]
    right = wcs.array_shape[1] - trim[1]
    bottom = 0 + trim[2]
    top = wcs.array_shape[0] - trim[3]
    xs = np.concatenate((
        np.arange(left, right),
        np.full(top-bottom, right - 1),
        np.arange(right - 1, left - 1, -1),
        np.full(top-bottom, left)))
    ys = np.concatenate((
        np.full(right - left, bottom),
        np.arange(bottom, top),
        np.full(right - left, top - 1),
        np.arange(top - 1, bottom - 1, -1)))
    
    ra, dec = wcs.all_pix2world(xs, ys, 0)
    assert not np.any(np.isnan(ra)) and not np.any(np.isnan(dec))
    ra_min = np.min(ra)
    ra_max = np.max(ra)
    # Heuristic to decide if we're straddling the wrap point
    if ra_min < 90 and ra_max > 270:
        ras1 = ra[ra < 180]
        ras2 = ra[ra > 180]
        ra_segments = [(np.min(ras1), np.max(ras1)),
                       (np.min(ras2), np.max(ras2))]
    else:
        ra_segments = [(ra_min, ra_max)]
    dec_min = np.min(dec)
    dec_max = np.max(dec)
    
    stars_ra, stars_dec, stars_vmag = list(zip(*star_catalog().stars_between(
        ra_segments, dec_min, dec_max)))
    stars_vmag = np.array(stars_vmag)
    stars_ra, stars_dec = np.array(stars_ra), np.array(stars_dec)
    
    # Filter to stars actually within the image
    stars_x, stars_y = wcs.all_world2pix(stars_ra, stars_dec, 0)
    filter = ((left < stars_x) * (stars_x < right)
              * (bottom < stars_y) * (stars_y < top))
    stars_x = stars_x[filter]
    stars_y = stars_y[filter]
    stars_vmag = stars_vmag[filter]
    stars_ra = stars_ra[filter]
    stars_dec = stars_dec[filter]
    
    # Apply magnitude cutoffs
    filter = (stars_vmag < dim_cutoff) * (stars_vmag > bright_cutoff)
    stars_x = stars_x[filter]
    stars_y = stars_y[filter]
    stars_vmag = stars_vmag[filter]
    stars_ra = stars_ra[filter]
    stars_dec = stars_dec[filter]
    
    return (stars_x, stars_y, stars_vmag, stars_ra, stars_dec)


class StarBins:
    """
    Class to allow efficient access to stars within an RA/Dec range
    
    Works by dividing RA/Dec space into bins and placing stars within a bin.
    Querying an RA/Dec range can then access only the relevant bins, cutting
    down the search space.
    """
    def __init__(self, RA_bin_size, dec_bin_size):
        
        self.n_RA_bins = int(ceil(360 / RA_bin_size))
        self.n_dec_bins = int(ceil(180 / dec_bin_size))
        
        self.bins = []
        for i in range(self.n_RA_bins):
            self.bins.append([[] for j in range(self.n_dec_bins)])
    
    def get_ra_bin(self, ra):
        ra = ra % 360
        ra_frac = ra / 360
        
        ra_bin = int(floor(ra_frac * self.n_RA_bins))
        
        return ra_bin
    
    def get_dec_bin(self, dec):
        dec_frac = (dec + 90) / 180
        
        dec_bin = int(floor(dec_frac * self.n_dec_bins))
        
        return dec_bin
    
    def get_bin(self, ra, dec):
        return self.get_ra_bin(ra), self.get_dec_bin(dec)
    
    def add_star(self, ra, dec, data):
        """
        Add a star to the bins
        
        Parameters
        ----------
        ra, dec : float
            The coordinates of the star
        data
            Any arbitrary object to be stored for this star
        """
        ra_bin, dec_bin = self.get_bin(ra, dec)
        
        self.bins[ra_bin][dec_bin].append(data)
    
    def get_stars(self, ra, dec):
        ra_bin, dec_bin = self.get_bin(ra, dec)
        
        return self.bins[ra_bin][dec_bin]
    
    def stars_between(self, ra_segments, dec_min, dec_max):
        """
        Generator to access stars within an RA/Dec range
        
        As a generator, it can be used like:
        
        for star_data in star_bins.stars_between(...):
            ...
        
        Parameters
        ----------
        ra_segments : list of tuples
            The segments in right ascension to access. To handle wrapping,
            multiple segments are supported. Each tuple in this list consists
            of (ra_start, ra_stop).
        dec_min, dec_max : float
            The declination range to search.
        """
        ra_seqs = []
        for ra_seg in ra_segments:
            bin_start = self.get_ra_bin(ra_seg[0])
            bin_end = self.get_ra_bin(ra_seg[1])
            ra_seqs.append(range(bin_start, bin_end+1))
        ra_bins = chain(*ra_seqs)
        bin_start = self.get_dec_bin(dec_min)
        bin_end = self.get_dec_bin(dec_max)
        dec_bins = range(bin_start, bin_end+1)
        
        for ra_bin in ra_bins:
            for dec_bin in dec_bins:
                yield from self.bins[ra_bin][dec_bin]


_star_catalog = None


def star_catalog():
    """Loads the stellar catalog bundled with this package"""
    # If the catalog has already been loaded and cached, use that
    global _star_catalog
    if _star_catalog is not None:
        return _star_catalog
    
    stars = StarBins(3, 3)
    
    catalog_path = utils.data_path("hipparchos_catalog.tsv")
    star_dat = open(catalog_path).readlines()
    for line in star_dat[43:-1]:
        try:
            id, RA, dec, Vmag = line.split(";")
        except ValueError:
            continue
        try:
            Vmag = float(Vmag)
        except ValueError:
            continue
        
        # Convert RA to floating-point degrees
        h, m, s = RA.split(" ")
        h = int(h) + int(m) / 60 + float(s) / 60 / 60
        RA = h / 24 * 360
        
        # Convert declination to floating-point degrees
        d, m, s = dec.split(" ")
        sign = 1 if d.startswith("+") else -1
        d = abs(int(d)) + int(m) / 60 + float(s) / 60 / 60
        dec = d * sign
        
        stars.add_star(RA, dec, (RA, dec, Vmag))
    
    _star_catalog = stars
    return stars