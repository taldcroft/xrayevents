import pyfits
import pywcs

from xrayevents import XrayEvents
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


    
