import numpy as np
import pytest

import remove_starfield
from .test_core import starfield


def test_starfield_save_load(starfield, tmp_path):
    save_path = tmp_path / "starfield"
    starfield.save(save_path)
    starfield2 = remove_starfield.Starfield.load(save_path)
    
    np.testing.assert_equal(starfield.starfield, starfield2.starfield)
    np.testing.assert_equal(starfield.attribution, starfield2.attribution)
    np.testing.assert_equal(starfield.frame_count, starfield2.frame_count)
    assert str(starfield.wcs) == str(starfield2.wcs)


@pytest.mark.array_compare(file_format='fits')
def test_starfield_subtract_from_image(starfield):
    test_file = remove_starfield.utils.test_data_path(
        'WISPR_files_preprocessed_quarter_size_L3',
        'psp_L3_wispr_20221206T223017_V1_1221.fits')
    
    subtracted = starfield.subtract_from_image(test_file)
    
    return np.stack((
        subtracted.blurred_data, subtracted.starfield_sample,
        subtracted.subtracted))
