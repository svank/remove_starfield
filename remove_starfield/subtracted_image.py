from dataclasses import dataclass

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from remove_starfield import utils


@dataclass
class SubtractedImage:
    source_file: str
    source_data: np.ndarray
    blurred_data: np.ndarray
    starfield_sample: np.ndarray
    wcs: WCS
    
    @property
    def subtracted(self):
        return self.blurred_data - self.starfield_sample
    
    def save(self, filename: str, overwrite=False):
        fits.writeto(filename, self.subtracted, self.wcs.to_header(),
                     overwrite=overwrite)
    
    def plot_comparison(self, pmin=1, pmax=99, **kwargs):
        vmin, vmax = np.nanpercentile(self.source_data, [pmin, pmax])

        fig, ax_locations = plt.subplots(2, 2, figsize=(12, 12),
                                sharex=True, sharey=True)
        
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
        
        def plot_subplot(ax, data):
            args = dict(origin='lower', cmap='Greys_r') | kwargs
            im = ax.imshow(data,
                    norm=matplotlib.colors.PowerNorm(
                        gamma=1/2.2, vmin=vmin, vmax=vmax),
                    **args)
        
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
        plot_subplot(ax, self.subtracted)
        ax.set_title("Input image minus starfield")
