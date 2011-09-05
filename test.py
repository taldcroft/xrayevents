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
img = evt.binned(x0=4000, x1=4100, binx=10, y0=4000, y1=4100, biny=10)


print img.header

    
dmimg = pyfits.open('t/acis_dm_bin10_img.fits.gz')[0]
dmheader = []
if 0:
    for key, val in dmimg.header.items():
        if key in ('COMMENT', 'HISTORY'):
            continue
        if re.match(r'[0-9]', key):
            continue
        if re.match(r'M|DS|BIAS|FP_TEMP|LIV|ONTIME|EXP|LT', key):
            continue
        if key in img.header:
            print key, val, img.header[key]
        else:
            print key, val, '--------'
            img.header.update(key, val)

        dmheader.append((key,val))

img.writeto('acis_bin10_py_img.fits', clobber=True)
