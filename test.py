import pyfits
import pywcs
import re
import xrayevents

import nose.tools as nt

def test_bin(bin=4):
    evts = XrayEvents('t/acis_evt2.fits.gz')
    print evts.pix2sky(4096.5, 4096.5)
    nt.assert_equal(1,2)
    
# evts = XrayEvents('t/acis_evt2.fits.gz')
# print evts.pix2sky(4096.5, 4096.5)


hdus = pyfits.open('t/acis_evt2.fits.gz')
header = hdus[1].header
wcs = pywcs.WCS(header=header, keysel=['pixel'], colsel=[11,12])
evt = xrayevents.XrayEvents('t/acis_evt2.fits.gz')

img = evt.image(x0=4000, x1=4100, binx=10, y0=4000, y1=4100, biny=10)
img.writeto('t/acis_bin10_py_img.fits', clobber=True)

img = evt.image(x0=4000, x1=4100, binx=1, y0=4000, y1=4100, biny=1)
img.writeto('t/acis_bin1_py_img.fits', clobber=True)

