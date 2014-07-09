import numpy as np
from astropy.io import fits
import xrayevents

# Some code for reference
# hdus = fits.open('t/acis_evt2.fits.gz')
# header = hdus[1].header
# wcs = pywcs.WCS(header=header, keysel=['pixel'], colsel=[11,12])


def test_image_bin1():
    """
    Compare to:
      dmcopy acis_evt2.fits.gz'[energy=2000:5000][bin x=4000:4100,y=4000:4100]' \
         acis_dm_bin1_img.fits.gz
    """
    evt = xrayevents.XrayEvents('t/acis_evt2.fits.gz')

    img = evt.image(x0=4000, x1=4100, binx=1, y0=4000, y1=4100, biny=1,
                    filters=[('energy', 2000., 5000.)])
    img.writeto('t/acis_py_bin1_img.fits', clobber=True)
    dmimg = fits.open('t/acis_dm_bin1_img.fits.gz')[0]
    assert np.all(img.data == dmimg.data)
    

def test_image_bin10():
    """
    Compare to:
      dmcopy acis_evt2.fits.gz'[bin x=4000:4100:10,y=4000:4100:10]' acis_dm_bin10_img.fits.gz
    """
    evt = xrayevents.XrayEvents('t/acis_evt2.fits.gz')

    img = evt.image(x0=4000, x1=4100, binx=10, y0=4000, y1=4100, biny=10)
    img.writeto('t/acis_py_bin10_img.fits', clobber=True)
    dmimg = fits.open('t/acis_dm_bin10_img.fits.gz')[0]
    assert np.all(img.data == dmimg.data)
