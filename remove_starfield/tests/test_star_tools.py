from .. import star_tools, utils

import numpy as np
import pytest
from pytest import approx


def test_load_stars(mocker):
    star_data = [''] * 43
    star_data.append('1;2 30 00;+10 30 00;2')
    star_data.append('2;2 30 30;+10 30 30;5')
    star_data.append('3;12 30 30;-10 30 30;10')
    star_data.append('4;8 00 00;-01 00 00;10')
    star_data.append('')
    
    class MockFile():
        def readlines(self):
            return star_data
    
    mocker.patch(star_tools.__name__+'.open',
            return_value=MockFile())
    
    try:
        star_cat = star_tools.star_catalog()
    finally:
        # Ensure this fake catalog doesn't get cached
        star_tools._star_catalog = None
    
    data = star_cat.get_stars(2.5/24*360, 10.5)
    assert len(data) == 2
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    
    data = star_cat.get_stars(12.4/24*360, -10.6)
    assert len(data) == 1
    assert data[0] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)
    
    data = star_cat.get_stars(0, 0)
    assert len(data) == 0
    
    data = list(star_cat.stars_between([(2/24*360, 3/24*360)], 10, 11))
    assert len(data) == 2
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    
    data = list(star_cat.stars_between(
        [(2/24*360, 3/24*360), (11/24*360, 13/24*360)], -11, 11))
    assert len(data) == 3
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    assert data[2] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)
    
    data = list(star_cat.stars_between(
        [(2/24*360, 13/24*360)], -11, 11))
    assert len(data) == 4
    assert data[0] == (2.5/24*360, 10.5, 2)
    assert data[1] == ((2.5 + .5/60)/24*360, 10.5 + .5/60, 5)
    assert data[2] == (8/24*360, -1, 10)
    assert data[3] == ((12.5 + .5/60)/24*360, -10.5 - .5/60, 10)


def test_extract_flux():
    # Create an image with three Gaussian stars
    xc = 6
    yc = 6
    x = np.arange(13) - xc
    y = np.arange(13) - yc
    x, y = np.meshgrid(x, y)

    width = 2
    star1 = np.exp(-x**2 / width**2 - y**2 / width**2)
    width = 3
    star2 = 2 * np.exp(-x**2 / width**2 - y**2 / width**2)

    star_cat = [
        (15, 22, star1),
        (32, 18, star2),
        (30, 36, star1)

    ]
    
    img = np.zeros((50, 50))
    for x, y, star in star_cat:
        ystart = int(y - star.shape[0]//2)
        ystop = ystart + star.shape[0]
        xstart = int(x - star.shape[1]//2)
        xstop = xstart + star.shape[1]
        img[ystart:ystop, xstart:xstop] = star
    
    # Create another image with a background offset
    img_with_bg = img + 20
    
    # Create another image with a random background
    np.random.seed(222)
    img_with_random_bg = img + .1 * np.random.random(img.shape)
    
    # Check that we measure correct fluxes
    for x, y, star in star_cat:
        fluxes = star_tools.extract_flux(
                [img, img_with_bg, img_with_random_bg],
                x, y, aperture_r=7, gap=2, annulus_thickness=3)
        assert fluxes[0] == approx(star.sum(), rel=1e-2)
        assert fluxes[1] == approx(star.sum(), rel=1e-2)
        assert fluxes[2] == approx(star.sum(), rel=5e-2)
    
    # Test skip_edge_stars
    fluxes = star_tools.extract_flux(
            [img, img_with_bg], 1, 1, aperture_r=7, skip_edge_stars=True)
    assert len(fluxes) == 0
    
    with pytest.raises(ValueError, match='Cutout does not fit in image'):
        fluxes = star_tools.extract_flux(
                [img, img_with_bg], 1, 1, aperture_r=7, skip_edge_stars=False)


@pytest.mark.array_compare
def test_find_expected_stars_in_frame():
    file = utils.test_data_path(
        'WISPR_files_with_data_half_size_L3', '20190405',
        'psp_L3_wispr_20190405T010554_V3_1221.fits')
    
    ret = star_tools.find_expected_stars_in_frame(file, trim=(10, 20, 30, 40))
    
    return np.array(ret)
