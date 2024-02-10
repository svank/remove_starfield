from ..processor import ImageHolder, ImageProcessor
from .. import utils


def test_Image_Holder():
    ih = ImageHolder(image='image', wcs='wcs')
    assert ih.image == 'image'
    assert ih.wcs == 'wcs'


def test_ImageProcessor_wcs_key():
    file = utils.test_data_path(
        'WISPR_files_with_data_half_size_L3', '20190405',
        'psp_L3_wispr_20190405T010554_V3_1221.fits')
    
    processor = ImageProcessor(wcs_key=' ')
    holder = processor.load_image(file)
    
    assert list(holder.wcs.wcs.ctype) == ['HPLN-ZPN', 'HPLT-ZPN']
    
    processor = ImageProcessor(wcs_key='A')
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
    
    post_img = processor.postprocess_image(holder.image, holder.wcs, holder)
    assert post_img is holder.image
    
    post_starfield = processor.postprocess_starfield_estimate(
        holder.image, holder)
    assert post_starfield is holder.image
    