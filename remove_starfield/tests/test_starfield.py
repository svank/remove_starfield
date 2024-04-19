import numpy as np

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
