import numpy as np
import pywcs
import pyfits

class XrayEvents(object):
    def __init__(self, filename, hdu=1):
        self.filename= filename
        hdus = pyfits.open(filename)
        self.header = hdus[hdu].header
        self.hdus = hdus
        events = hdus[hdu].data
        self.events = events[np.argsort(events['x'])]
        self.wcs = pywcs.WCS(self.header, keysel=['pixel'],
                             colsel=[self._radec_col('RA'), self.radec_col('DEC')])

    def _radec_col(self, radec):
        """Return the column number corresponding to the RA or DEC coordinate.
        """
        for key, val in self.header.items():
            if val == radec + '---TAN':
                return key[5:]
        else:
            raise ValueError('No RA---TAN ctype found')

    def pix2sky(self, i, j):
        return self.wcs.wcs_pix2sky([[i, j]], 1)[0]

    def sky2pix(self, x, y):
        return self.wcs.wcs_sky2pix([[x, y]], 1)[0]

    def filtered(self, filters=None):
        if not filters:
            return self.events

        ok = np.ones(len(events), dtype=np.bool)
        for colname, limits in filters:
            colvals = events[colname]
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


    def binned(self,
               x0=None, x1=None, binx=1.0,
               y0=None, y1=None, biny=1.0,
               filters=None):
        dx = float(dx)
        dy = float(dy)

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
        events = self.event_filter(filters, events[ok])

        img, x_bins, y_bins = np.histogram2d(events['y'], events['x'],
                                             bins=[np.arange(y0, y1, dy),
                                                   np.arange(x0, x1, dx)])

        # Find the position in image coords of the sky pix reference position
        # The -0.5 assumes that image coords refer to the center of the image bin.
        x_crpix = (self.wcs.wcs.crpix[0] - (x0 - dx / 2.0)) / dx
        y_crpix = (self.wcs.wcs.crpix[1] - (y0 - dy / 2.0)) / dy

        # Create the image => sky transformation
        wcs = pywcs.WCS(naxis=2)
        wcs.wcs.equinox = 2000.0
        wcs.wcs.crpix = [x_crpix, y_crpix]
        wcs.wcs.cdelt = [self.wcs.wcs.cdelt[0] * dx, self.wcs.wcs.cdelt[1] * dy]
        wcs.wcs.cunit = [self.wcs.wcs.cunit[0], self.wcs.wcs.cunit[1]]
        wcs.wcs.crval = [self.wcs.wcs.crval[0], self.wcs.wcs.crval[1]]
        wcs.wcs.ctype = [self.wcs.wcs.ctype[0], self.wcs.wcs.ctype[1]]
        header = wcs.to_header()

        # Create the image => physical transformation and add to header
        wcs = pywcs.WCS(naxis=2)
        wcs.wcs.crpix = [0.5, 0.5]
        wcs.wcs.cdelt = [dx, dy]
        wcs.wcs.crval = [x0, y0]
        wcs.wcs.ctype = ['x', 'y']

        for key, val in wcs.to_header().items():
            header.update(key + 'P', val)
        header.update('WCSTY1P', 'PHYSICAL')
        header.update('WCSTY2P', 'PHYSICAL')

        hdu = pyfits.PrimaryHDU(np.array(img, dtype=np.int32), header=header)
        return hdu

