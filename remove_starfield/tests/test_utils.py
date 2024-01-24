from copy import deepcopy

from astropy.wcs import WCS
import pytest

from .. import utils

@pytest.mark.parametrize('wcs_key', [' ', 'A'])
def test_find_bounds(wcs_key):
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 1, 1
    wcs_out.wcs.crval = 0, 0
    wcs_out.wcs.cdelt = 1, 1
    wcs_out.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 1, 1
    # Add an offset to avoid anything landing right at pixel boundaries and so
    # having to care about floating-point error
    wcs_in.wcs.crval = 0.1, 0.1
    wcs_in.wcs.cdelt = 1, 1
    wcs_in.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    header = wcs_in.to_header(key=wcs_key)
    header['NAXIS1'] = 10
    header['NAXIS2'] = 12
    
    bounds = utils.find_bounds(header, wcs_out, key=wcs_key)
    
    assert bounds == (0, 10, 0, 12)
    
    bounds = utils.find_bounds(header, wcs_out, trim=(1,2,4,5),
            key=wcs_key)
    
    assert bounds == (1, 8, 4, 7)


def test_find_bounds_coord_bounds():
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 1, 1
    wcs_out.wcs.crval = 0, 0
    wcs_out.wcs.cdelt = 1, 1
    wcs_out.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 1, 1
    # Add an offset to avoid anything landing right at pixel boundaries and so
    # having to care about floating-point error
    wcs_in.wcs.crval = 0.1, 0.1
    wcs_in.wcs.cdelt = 1, 1
    wcs_in.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    header = wcs_in.to_header()
    header['NAXIS1'] = 10
    header['NAXIS2'] = 12
    
    bounds = utils.find_bounds(
            header, wcs_out, world_coord_bounds=[1, 5, 2, 6])
    
    assert bounds is None
    
    bounds = utils.find_bounds(
            header, wcs_out, world_coord_bounds=[1, 5, None, None])
    
    assert bounds == (1, 5, 0, 12)
    
    bounds = utils.find_bounds(
            header, wcs_out, world_coord_bounds=[None, None, 2, 6])
    
    assert bounds == (0, 10, 2, 6)
    
    bounds = utils.find_bounds(
            header, wcs_out, world_coord_bounds=[None, 4, None, 6])
    
    assert bounds == (0, 4, 0, 6)


@pytest.mark.parametrize('wcs_key', [' ', 'A'])
def test_find_collective_bounds(wcs_key):
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = 1, 1
    wcs_out.wcs.crval = 0, 0
    wcs_out.wcs.cdelt = 1, 1
    wcs_out.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = 1, 1
    # Add an offset to avoid anything landing right at pixel boundaries and so
    # having to care about floating-point error
    wcs_in.wcs.crval = 0.1, 0.1
    wcs_in.wcs.cdelt = 1, 1
    wcs_in.wcs.ctype = "HPLN-CAR", "HPLT-CAR"
    
    header1 = wcs_in.to_header(key=wcs_key)
    header1['NAXIS1'] = 10
    header1['NAXIS2'] = 12
    
    header2 = deepcopy(header1)
    header2['NAXIS2'] = 24
    
    bounds = utils.find_collective_bounds([header1, header2], wcs_out,
            key=wcs_key)
    # Bounds are (0, 10, 0, 12) and (0, 10, 0, 24)
    assert bounds == (0, 10, 0, 24)
    
    header2['CRPIX1' + wcs_key] = 2
    header2['CRPIX2' + wcs_key] = 3
    
    bounds = utils.find_collective_bounds([header1, header2], wcs_out,
            key=wcs_key)
    # Bounds are (0, 10, 0, 12) and (-1, 9, -2, 22)
    assert bounds == (-1, 10, -2, 22)
    
    # Test `trim` values
    bounds = utils.find_collective_bounds([header1, header2], wcs_out,
            trim=(1, 2, 4, 5), key=wcs_key)
    # Bounds are (1, 8, 4, 7) and (0, 7, 2, 17)
    assert bounds == (0, 8, 2, 17)
    
    # Test multiple sub-lists and only one trim value to apply to each
    bounds = utils.find_collective_bounds([[header1], [header2]], wcs_out,
            trim=(1, 2, 4, 5), key=wcs_key)
    # Bounds are (1, 8, 4, 7) and (0, 7, 2, 17)
    assert bounds == (0, 8, 2, 17)
    
    # Test multiple sub-lists and a separate trim value to apply to each
    bounds = utils.find_collective_bounds([[header1], [header2]], wcs_out,
            trim=[(0, 0, 0, 0), (1, 2, 4, 5)], key=wcs_key)
    # Bounds are (0, 10, 0, 12) and (0, 7, 2, 17)
    assert bounds == (0, 10, 0, 17)
    
    bounds = utils.find_collective_bounds([[header1], [header2]], wcs_out,
            trim=[(1, 2, 4, 5), (0, 0, 0, 0)], key=wcs_key)
    # Bounds are (1, 8, 4, 7) and (-1, 9, -2, 22)
    assert bounds == (-1, 9, -2, 22)
    
    # Finally test just one header
    bounds = utils.find_collective_bounds([header1], wcs_out,
            trim=(1, 2, 4, 5), key=wcs_key)
    assert bounds == (1, 8, 4, 7)
    bounds = utils.find_collective_bounds(header1, wcs_out,
            trim=(1, 2, 4, 5), key=wcs_key)
    assert bounds == (1, 8, 4, 7)
