from dataclasses import dataclass, field

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from . import utils
import remove_starfield


@dataclass
class SubtractedImage:
    """Container class for a subtracted image and associated data
    """
    subtracted: np.ndarray = field(init=False)
    """The blurred source data with the starfield estimate removed"""
    wcs: WCS
    """The WCS which describes all of the data arrays"""
    meta: dict | fits.Header
    """The header information from the input file"""
    source_file: str
    """The file path from which the image-to-be-subtracted was loaded"""
    source_data: np.ndarray
    """The data loaded from the source file"""
    blurred_data: np.ndarray
    """The source data after being blurred (to match the blurring inherent in
        the anti-aliased reprojection used to build the starfield estimate, and
        then to project that estimate into the source image's frame). When
        comparing the post-subtraction image to the source image, this is
        likely the right source image to compare to."""
    starfield_sample: np.ndarray
    """The starfield estimate for this image"""
    
    def __post_init__(self):
        self.subtracted = self.blurred_data - self.starfield_sample
    
    def save(self, filename: str, overwrite=False):
        """Writes the subtracted image to a FITS file, with WCS header included

        Parameters
        ----------
        filename : ``str``
            The location to which the FITS file should be written
        overwrite : ``bool``, optional
            Whether to overwrite any existing file at the destination path, by
            default False.
        """
        header = fits.Header(self.meta)
        header['HISTORY'] = ("Starfield subtracted by remove_starfield "
                             f"{remove_starfield.__version__}")
        fits.writeto(filename, self.subtracted, header,
                     overwrite=overwrite)
    
    def plot_comparison(self, vmin='auto', vmax='auto', pmin=1, pmax=99,
                        bwr=False, **kwargs):
        """Produces a 2x2 array of plots for easy evaluation

        Parameters
        ----------
        vmin, vmax : ``float``, optional
            Colormap range. If either is ``'auto'``, use a percentile value,
            set by ``pmin/pmax``.
        pmin, pmax : float, optional
            The percentile values to use if ``vmin/vmax`` is ``'auto'``.
        bwr : ``bool``, optional
            If True, instead of using a "space image" colormap with black
            background and white stars, uses a blue-white-red colormap, where
            white is the median value of the input image (to estimate the
            typical background level). The colormap extends equal amounts above
            and below this center (the greater of vmin and vmax's distances
            from the center). The gamma factor of the colormap is set to 1
            instead of the usual 1/2.2. This produces a very symmetric
            representation of over- and under-subtractions of stars (in blue
            and red, respectively). While the default gray colormap is good for
            presenting the inputs and outputs normally, the clipping of the
            colorbar on the low side may over-emphasize the appearance of
            oversubtractions (as black "holes" in the image when their low
            values are clipped), whereas under-subtractions aren't clipped and
            so more easily blend in to the texture or noise of the image.
        **kwargs
            Passed to the ``plt.imshow`` call
        """
        if 'auto' in (vmin, vmax):
            pmin, pmax = np.nanpercentile(self.blurred_data, [pmin, pmax])
            vmin = pmin if vmin == 'auto' else vmin
            vmax = pmax if vmax == 'auto' else vmax

        fig, ax_locations = plt.subplots(
            2, 2, figsize=(12, 12), sharex=True, sharey=True)
        
        ax_locations = ax_locations.flatten()
        axs = []
        for ax in ax_locations:
            position = ax.get_position().bounds
            ax.remove()
            ax = WCSAxes(fig, position, wcs=self.wcs,
                         sharex=axs[-1] if len(axs) else None,
                         sharey=axs[-1] if len(axs) else None)
            fig.add_axes(ax)
            axs.append(ax)
            utils.prepare_axes(ax)
        
        if bwr:
            cmap = 'bwr'
            center = np.nanmedian(self.blurred_data)
            ex = max(np.abs([vmax - center, vmin - center]))
            vmin = center - ex
            vmax = center + ex
            gamma = 1
        else:
            cmap = 'Greys_r'
            gamma = 1/2.2
        
        def plot_subplot(ax, data):
            lon, lat = ax.coords
            lon.set_axislabel("Right Ascension")
            lat.set_axislabel("Declination")
            args = dict(origin='lower', cmap=cmap) | kwargs
            im = ax.imshow(data,
                    norm=matplotlib.colors.PowerNorm(
                        gamma=gamma, vmin=vmin, vmax=vmax),
                    **args)
            return im
        
        ax = axs[0]
        plot_subplot(ax, self.source_data)
        ax.set_title("Input image")

        ax = axs[1]
        plot_subplot(ax, self.blurred_data)
        ax.set_title("Blurred input image")

        ax = axs[2]
        plot_subplot(ax, self.starfield_sample)
        ax.set_title("Starfield estimate")

        ax = axs[3]
        im = plot_subplot(ax, self.subtracted)
        ax.set_title("Input image minus starfield")
        
        # Set this image to be the one found by plt.colorbar, for instance. But
        # if this manager attribute is empty, pyplot won't accept it.
        if ax.figure.canvas.manager:
            plt.sca(ax)
            plt.sci(im)
