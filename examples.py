import numpy as np
import pyfits
import pywcs
import xrayevents

# Some code for reference
# hdus = pyfits.open('t/acis_evt2.fits.gz')
# header = hdus[1].header
# wcs = pywcs.WCS(header=header, keysel=['pixel'], colsel=[11,12])

#
# dmcopy acis_evt2.fits.gz'[bin x=4000:4100:10,y=4000:4100:10]' \
#        acis_dm_bin10_img.fits.gz

evt = xrayevents.XrayEvents('t/acis_evt2.fits.gz')

img = evt.image(x0=4000, x1=4100, binx=10, y0=4000, y1=4100, biny=10)
img.writeto('bin10_img.fits', clobber=True)

print img.header
print img.data

dmimg = pyfits.open('t/acis_dm_bin10_img.fits.gz')[0]
print dmimg.header
print dmimg.data

ra, dec = evt.pix2sky(3900, 4250)
print ra, dec

x, y = evt.sky2pix(ra, dec)
print x, y


# Emulate dmcopy command
# dmcopy acis_evt2.fits.gz'[energy=2000:5000][bin x=4000:4100:10,y=4000:4100:10]' \
#        acis_dm_bin1_img.fits.gz

img_hard = evt.image(x0=4000, x1=4100, binx=10, y0=4000, y1=4100, biny=10,
                     filters=[('energy', 2000., 7000.)])
print img_hard.data
    
# Use pyregion

import pyregion
regions = pyregion.open('t/ds9.reg')

for region in regions:
    print region, "\n"
    
print regions[0].name
print regions[0].coord_format
print regions[0].coord_list

