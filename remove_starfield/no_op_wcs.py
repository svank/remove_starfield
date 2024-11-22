from astropy.wcs.wcsapi import BaseLowLevelWCS, HighLevelWCSWrapper


class NoOpWCS(BaseLowLevelWCS):
    """
    A fast option for when using pixel_to_pixel with the same in and out WCS.
    
    We do that to provide compatible blurring to input data before
    subtraction. If we use the actual image's WCS for the input and output
    projections, we have to do a lot of coordinate computations for nothing.
    This "WCS" avoids that.
    """
    def __init__(self, input_wcs, input_data_array=None):
        self.input_wcs = input_wcs
        if input_data_array is None:
            self.input_data_pixel_shape = input_wcs.pixel_shape[-2:]
        else:
            self.input_data_pixel_shape = (input_data_array.shape[-1],
                                           input_data_array.shape[-2])
    
    def pixel_to_world_values(self, *pixel_arrays):
        """ """
        result = []
        # We would just output the pixel indices as the world coordinate,
        # but that can produce illegal angles (like latitude > 90 deg),
        # so we instead scale the pixel coordinates at a 0-1 range.
        for pixel_array, n_values in zip(pixel_arrays, self.pixel_shape):
            result.append(pixel_array / n_values)
        return result
    
    # The empty docstrings suppress the docs pulling in the base class
    # docstrings
    def world_to_pixel_values(self, *world_arrays):
        """ """
        result = []
        # Here we undo the scaling from pixel_to_world_values
        for world_array, n_values in zip(world_arrays, self.pixel_shape):
            result.append(world_array * n_values)
        return result
    
    @property
    def pixel_n_dim(self):
        """ """
        return self.input_wcs.pixel_n_dim
    
    @property
    def world_n_dim(self):
        """ """
        return self.input_wcs.world_n_dim
    
    @property
    def pixel_shape(self):
        return self.input_data_pixel_shape or self.input_wcs.pixel_shape
    
    @property
    def world_axis_units(self):
        """ """
        return self.input_wcs.world_axis_units
    
    @property
    def world_axis_physical_types(self):
        """ """
        return self.input_wcs.world_axis_physical_types
    
    @property
    def world_axis_object_components(self):
        """ """
        return self.input_wcs.world_axis_object_components
    
    @property
    def world_axis_object_classes(self):
        """ """
        return self.input_wcs.world_axis_object_classes
    
    def as_high_level(self):
        """ """
        return HighLevelWCSWrapper(self)
