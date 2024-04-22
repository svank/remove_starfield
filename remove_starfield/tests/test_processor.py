from ..processor import ImageHolder, ImageProcessor
from .. import utils


def test_Image_Holder():
    ih = ImageHolder(data='image', wcs='wcs', meta='meta')
    assert ih.data == 'image'
    assert ih.wcs == 'wcs'
    assert ih.meta == 'meta'


def test_ImageProcessor_wcs_key():
    file = utils.test_data_path(
        'WISPR_files_with_data_half_size_L3', '20190405',
        'psp_L3_wispr_20190405T010554_V3_1221.fits')
    
    processor = ImageProcessor()
    holder = processor.load_image(file)
    
    assert list(holder.wcs.wcs.ctype) == ['RA---ZPN', 'DEC--ZPN']


def test_ImageProcessor_passthroughs():
    file = utils.test_data_path(
        'WISPR_files_with_data_half_size_L3', '20190405',
        'psp_L3_wispr_20190405T010554_V3_1221.fits')
    
    processor = ImageProcessor()
    holder = processor.load_image(file)
    holder2 = processor.preprocess_image(holder)
    assert holder2 is holder
    
    post_img = processor.postprocess_image(holder.data, holder.wcs, holder)
    assert post_img is holder.data
    
    post_starfield = processor.postprocess_starfield_estimate(
        holder.data, holder)
    assert post_starfield is holder.data
    