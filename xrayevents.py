import numpy as np
import pywcs
import pyfits

def event_filter(filters, events=None):
    if events is None:
        events = self.events
    if not filters:
        return events

    ok = np.ones(len(events), dtype=np.bool)
    for colname, limits in filters:
        colvals = events.field(colname)
        try:
            lo, hi = limits
            if lo is None and hi is not None:
                ok &= (colvals < hi)
            elif lo is not None and hi is None:
                ok &= (colvals >= lo)
            elif lo is not None and hi is not None:
                ok &= (colvals >= lo) & (colvals < hi)
        except TypeError:
            ok &= (colvals == limits)

    return events[ok]

class XrayEvents(object):
    def __init__(self, filename, hdu=1):
        self.filename= filename
        hdus = pyfits.open(filename)
        self.header = hdus[hdu].header
        self.hdus = hdus
        events = hdus[hdu].data
        self.events = events[np.argsort(events['x'])]
        self.wcs = pywcs.WCS(self.header, keysel=['pixel'],
                             colsel=[self._find_col('RA---TAN'), self._find_col('DEC--TAN')])

    def _find_col(self, hdrkey):
        """Return the column number corresponding to the RA or DEC coordinate.
        """
        for key, val in self.header.items():
            if val == hdrkey:
                return key[5:]
        else:
            raise ValueError('No RA---TAN ctype found')

    def pix2sky(self, i, j):
        return self.wcs.wcs_pix2sky([[i, j]], 1)[0]

    def sky2pix(self, x, y):
        return self.wcs.wcs_sky2pix([[x, y]], 1)[0]

    def binned(self,
               x0=None, x1=None, binx=1.0,
               y0=None, y1=None, biny=1.0,
               filters=None):
        binx = float(binx)
        biny = float(biny)

        events = self.events
        if x0 is None:
            x0 = np.min(events['x'])
        if x1 is None:
            x1 = np.max(events['x'])
        if y0 is None:
            y0 = np.min(events['y'])
        if y1 is None:
            y1 = np.max(events['y'])

        i0, i1 = np.searchsorted(events['x'], [x0, x1])
        events = events[i0:i1]
        ok = (events['y'] >= y0) & (events['y'] <= y1)
        events = event_filter(filters, events[ok])

        img, x_bins, y_bins = np.histogram2d(events['y'], events['x'],
                                             bins=[np.arange(y0, y1, biny),
                                                   np.arange(x0, x1, binx)])

        # Find the position in image coords of the sky pix reference position
        # The -0.5 assumes that image coords refer to the center of the image bin.
        x_crpix = (self.wcs.wcs.crpix[0] - (x0 - binx / 2.0)) / binx
        y_crpix = (self.wcs.wcs.crpix[1] - (y0 - biny / 2.0)) / biny

        # Create the image => sky transformation
        wcs = pywcs.WCS(naxis=2)
        wcs.wcs.equinox = 2000.0
        wcs.wcs.crpix = [x_crpix, y_crpix]
        wcs.wcs.cdelt = [self.wcs.wcs.cdelt[0] * binx, self.wcs.wcs.cdelt[1] * biny]
        wcs.wcs.cunit = [self.wcs.wcs.cunit[0], self.wcs.wcs.cunit[1]]
        wcs.wcs.crval = [self.wcs.wcs.crval[0], self.wcs.wcs.crval[1]]
        wcs.wcs.ctype = [self.wcs.wcs.ctype[0], self.wcs.wcs.ctype[1]]
        header = wcs.to_header()

        # Create the image => physical transformation and add to header
        wcs = pywcs.WCS(naxis=2)
        wcs.wcs.crpix = [0.5, 0.5]
        wcs.wcs.cdelt = [binx, biny]
        wcs.wcs.crval = [x0, y0]
        wcs.wcs.ctype = ['x', 'y']

        for key, val in wcs.to_header().items():
            header.update(key + 'P', val)
        header.update('WCSTY1P', 'PHYSICAL')
        header.update('WCSTY2P', 'PHYSICAL')

        # Set LTVi and LTMi_i keywords (seems to be needed for ds9)
        imgx0, imgy0 = wcs.wcs_sky2pix([[0.0, 0.0]], 1)[0]
        imgx1, imgy1 = wcs.wcs_sky2pix([[1.0, 1.0]], 1)[0]
        header.update('LTM1_1', imgx1 - imgx0)
        header.update('LTM2_2', imgy1 - imgy0)
        header.update('LTV1', imgx0)
        header.update('LTV2', imgy0)

        hdu = pyfits.PrimaryHDU(np.array(img, dtype=np.int32), header=header)
        return hdu

