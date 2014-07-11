from __future__ import division

import numpy as np
from astropy import wcs
from astropy.io import fits


def _arange_inclusive(x0, x1, binx):
    """
    Return np.arange(x0, x1, binx) except that range is inclusive of x1.
    """
    delx = (x1 - x0)
    nbin = delx / binx
    if abs(round(nbin) - nbin) < 1e-8:
        return np.linspace(x0, x1, round(nbin) + 1)
    else:
        return np.arange(x0, x1, binx, dtype=np.float)


def get_event_hdu_wcs(hdus, hdu_num=None):
    """
    Get the event list table and corresponding WCS in ``hdus``

    Raises TypeError if the ``hdus`` do not contain a valid event list with
    defined WCS info.

    :param hdus: FITS HDU list object
    :param hdu_num: HDU number (default=first matching HDU)
    :returns hdu, wcs: tuple of HDU and corresponding WCS object.
    """
    if hdu_num is None:
        hdus = hdus[1:]
    else:
        hdus = [hdus[hdu_num]]

    for hdu in hdus:
        colnames = [col.name.lower() for col in hdu.columns]
        try:
            x_idx = colnames.index('x') + 1
            y_idx = colnames.index('y') + 1
            w = wcs.WCS(hdu.header, keysel=['pixel'], colsel=(x_idx, y_idx))
        except:
            pass
        else:
            return hdu, w
    else:
        raise TypeError('FITS file has no event table extensions')


def event_filter(events, filters):
    """
    Filter ``events`` based on matching or limits on event columns.

    ``filters`` must be a list of tuples either 2 or 3 elements long:

    - (col_name, value)
    - (col_name, low_value | None, high_value | None)

    :param filters: list of tuples defining filters

    :returns: filtered table of events
    """
    if not filters:
        return events

    ok = np.ones(len(events), dtype=np.bool)
    for filter_ in filters:
        colname = filter_[0]
        colvals = events[colname]
        if len(filter_) == 2:
            ok &= (colvals == filter_[1])
        elif len(filter_) == 3:
            lo, hi = filter_[1], filter_[2]
            if lo is None and hi is not None:
                ok &= (colvals < hi)
            elif lo is not None and hi is None:
                ok &= (colvals >= lo)
            elif lo is not None and hi is not None:
                ok &= (colvals >= lo) & (colvals < hi)
            else:
                raise ValueError('Filter must contain either 2 or 3 values')

    return events[ok]


class FITSEventList(object):
    def __init__(self, hdus, hdu_num=None):
        """
        Object containing an X-ray event file with WCS and binning convenience methods

        :hdus: FITS HDU list object containing an event data table
        :param hdu_num: HDU number (default=first matching HDU)
        """
        self.event_hdu, self.wcs = get_event_hdu_wcs(hdus, hdu_num)
        self.header = self.event_hdu.header
        events = self.event_hdu.data
        events_x = events['x']
        self.events = events[np.argsort(events_x)]

    def pix2world(self, x, y):
        """
        Get world coordinates for (x, y)

        :param x: Pixel x value
        :param y: Pixel y value

        :returns: World coordinate (ra, dec) values
        """
        return self.wcs.wcs_pix2world([[x, y]], 1)[0]

    def world2pix(self, ra, dec):
        """
        Get pixel coordinates for (ra, dec)

        :param ra: World ra value
        :param dec: World dec value

        :returns: pixel coordinate (x, y) values
        """
        return self.wcs.wcs_world2pix([[ra, dec]], 1)[0]

    def image(self, x0=None, x1=None, binx=1.0, y0=None, y1=None, biny=1.0,
              filters=None, dtype=np.int32):
        """
        Create a binned image corresponding to the X-ray event (x, y) pairs.

        :param x0: lower limit of x (default = min(x))
        :param x1: upper limit of x (default = max(x))
        :param binx: bin size in x
        :param y0: lower limit of y (default = min(y))
        :param y1: upper limit of y (default = max(y))
        :param biny: bin size in y
        :param filters: table filters to apply using ``event_filters()``
        :param dytpe: output image array dtype

        :returns: fits.PrimaryHDU object with binned image
        """

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
        events = event_filter(events[ok], filters)

        x_bins = _arange_inclusive(x0, x1, binx)
        y_bins = _arange_inclusive(y0, y1, biny)
        if len(events) > 0:
            # Bug in np.histogram2d as of July 2011
            # http://old.nabble.com/histogram2d-error-with-empty-inputs-td31940769.html
            img, x_bins, y_bins = np.histogram2d(events['y'], events['x'],
                                                 bins=[y_bins, x_bins])
        else:
            img = np.zeros((len(x_bins) - 1, len(y_bins) - 1))

        # Find the position in image coords of the sky pix reference position
        # The -0.5 assumes that image coords refer to the center of the image bin.
        x_crpix = (self.wcs.wcs.crpix[0] - (x0 - binx / 2.0)) / binx
        y_crpix = (self.wcs.wcs.crpix[1] - (y0 - biny / 2.0)) / biny

        # Create the image => sky transformation
        w = wcs.WCS(naxis=2)
        w.wcs.equinox = 2000.0
        w.wcs.crpix = [x_crpix, y_crpix]
        w.wcs.cdelt = [self.wcs.wcs.cdelt[0] * binx, self.wcs.wcs.cdelt[1] * biny]
        w.wcs.cunit = [self.wcs.wcs.cunit[0], self.wcs.wcs.cunit[1]]
        w.wcs.crval = [self.wcs.wcs.crval[0], self.wcs.wcs.crval[1]]
        w.wcs.ctype = [self.wcs.wcs.ctype[0], self.wcs.wcs.ctype[1]]
        header = w.to_header()

        # Create the image => physical transformation and add to header
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [0.5, 0.5]
        w.wcs.cdelt = [binx, biny]
        w.wcs.crval = [x0, y0]
        w.wcs.ctype = ['x', 'y']

        for key, val in w.to_header().items():
            header[key + 'P'] = val
        header['WCSTY1P'] = 'PHYSICAL'
        header['WCSTY2P'] = 'PHYSICAL'

        # Set LTVi and LTMi_i keywords (seems to be needed for ds9)
        imgx0, imgy0 = w.wcs_world2pix([[0.0, 0.0]], 1)[0]
        imgx1, imgy1 = w.wcs_world2pix([[1.0, 1.0]], 1)[0]
        header['LTM1_1'] = imgx1 - imgx0
        header['LTM2_2'] = imgy1 - imgy0
        header['LTV1'] = imgx0
        header['LTV2'] = imgy0

        hdu = fits.PrimaryHDU(np.array(img, dtype=dtype), header=header)
        return hdu


class XrayEvents(FITSEventList):
    def __init__(self, filename, hdu=None):
        """
        Object containing an X-ray event file with WCS and binning convenience methods.
        Legacy version that accepts a filename.

        :param filename: event FITS file
        :hdu: HDU number containing the event data table (default=first event table)
        """
        self.filename = filename
        hdus = fits.open(filename)
        super(XrayEvents, self).__init__(hdus, hdu)
        hdus.close()
