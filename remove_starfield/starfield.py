import copy
from dataclasses import dataclass

from astropy.wcs import WCS
import h5py
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import reproject

from . import ImageProcessor, SubtractedImage, utils
from .processor import ImageHolder


@dataclass
class Starfield:
    """Class representing an estimated, all-sky background map"""
    starfield: np.ndarray
    """The all-sky map"""
    wcs: WCS
    """A WCS describing the map"""
    frame_count: np.ndarray | None = None
    """The number of input images that contributed to each pixel in the map"""
    attribution: np.ndarray | None = None
    """The (approximate) source file number of the value in each pixel"""
    
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
            f.create_dataset("starfield", data=self.starfield,
                             compression="gzip", shuffle=True)
            f.create_dataset("wcs", data=self.wcs.to_header_string())
            if self.frame_count is not None:
                f.create_dataset("frame_count", data=self.frame_count,
                                 compression="gzip", shuffle=True)
            if self.attribution is not None:
                f.create_dataset("attribution", data=self.attribution,
                                 compression="gzip", shuffle=True)
    
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
             grid=False, use_wcs=True, **kwargs):
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
        use_wcs : ``bool``
            Whether to plot in world coordinates instead of pixel coordinates.
        **kwargs
            Passed to the ``plt.imshow()`` call.

        Returns
        -------
        im
            The return value from the ``plt.imshow`` call
        """
        ax = utils.prepare_axes(ax, self.wcs if use_wcs else None, grid)
        
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
                         use_wcs=True, **kwargs):
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
        use_wcs : ``bool``
            Whether to plot in world coordinates instead of pixel coordinates.
        **kwargs
            Passed to the ``plt.imshow()`` call.

        Returns
        -------
        im
            The return value from the ``plt.imshow`` call
        """
        if self.frame_count is None:
            raise ValueError("This Starfield doesn't have a frame_count array")
        
        ax = utils.prepare_axes(ax, self.wcs if use_wcs else None, grid)
        
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
                         mapper=None, use_wcs=True, **kwargs):
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
        use_wcs : ``bool``
            Whether to plot in world coordinates instead of pixel coordinates.
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
        
        ax = utils.prepare_axes(ax, self.wcs if use_wcs else None, grid)
        
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
    
    def subtract_from_image(self,
                            file: str | ImageHolder,
                            processor: ImageProcessor=ImageProcessor()):
        """Subtracts this starfield from an image
        
        The provided image file is loaded and pre-processed, and then this
        `Starfield` is projected into the frame of the input image. Before
        subtraction, the input image receives a Gaussian blurring to match that
        which is inherently present in the starfield estimate.

        Parameters
        ----------
        file : ``str`` or ImageHolder or object with data and wcs attrs
            The input file from which to remove stars
        processor : `ImageProcessor`, optional
            An `ImageProcessor` to load the file and pre-process the file.

        Returns
        -------
        subtracted : `SubtractedImage`
            A container class storing the subtracted image and all inputs
        """
        if isinstance(file, str):
            image_holder = processor.load_image(file)
        elif hasattr(file, "wcs") and hasattr(file, "data"):
            # it's like an ImageHolder
            image_holder = file
        else:
            raise TypeError("Input file must be a str or an object with `data` "
                            "and `wcs` attrs")

        image_holder = processor.preprocess_image(image_holder)
        input_data = image_holder.data
        input_wcs = image_holder.wcs
        
        # Project the starfield into the input image's frame
        starfield_sample = reproject.reproject_adaptive(
            (self.starfield, self.wcs), input_wcs, input_data.shape,
            roundtrip_coords=False, return_footprint=False, x_cyclic=True,
            conserve_flux=True, center_jacobian=True, despike_jacobian=True)
        
        starfield_sample = processor.postprocess_starfield_estimate(
            starfield_sample, image_holder)
        
        # Ensure the input data receives the same Gaussian blurring as the
        # starfield data has (once in the reprojection to build the starfield
        # estimate, again in the reprojection back to this input image's frame)
        
        # TODO: Figure out if this is exactly correct, and if we can combine
        # these two blurs into one round of blurring
        img_r = reproject.reproject_adaptive(
            (input_data, input_wcs), input_wcs, input_data.shape,
            roundtrip_coords=False, return_footprint=False, conserve_flux=True,
            boundary_mode='ignore')
        img_r = reproject.reproject_adaptive(
            (img_r, input_wcs), input_wcs, input_data.shape,
            roundtrip_coords=False, return_footprint=False, conserve_flux=True,
            boundary_mode='ignore')
        
        return SubtractedImage(
            source_file=file,
            source_data=input_data,
            starfield_sample=starfield_sample,
            blurred_data=img_r,
            wcs=input_wcs,
            meta=image_holder.meta,
        )
