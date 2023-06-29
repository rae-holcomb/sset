### the Field class
# Library of functions to be used when dealing with TRC data
import numpy as np
import lightkurve as lk
import math
import matplotlib.pyplot as plt
import copy
from scipy import stats


import PRF

import astropy
from astroquery.mast import Catalogs
import astropy.units as u
import astropy.wcs as wcs
# from astropy.stats import sigma_clip
# from scipy import ndimage
import typing

import trc_funcs as trc

class Field():
    """Creates an array that 
    Inputs
        noise_func - the function used to be 
        noice_coeffs - the coefficients for the noise model"""
    def __init__(self, orig_tpf: lk.TessTargetPixelFile,
        source_cat: astropy.table,
        bkg_polyorder: int, 
        noise_func: typing.Callable=None, 
        noise_coeffs: np.array=None,
        pos_time=None, pos_corr1=None, pos_corr2=None,
        id: int=None):

        # define some useful variables
        self.orig_tpf = orig_tpf
        self.source_cat = source_cat
        self.shape = orig_tpf.shape[1:]
        # self.field = np.zeros(self.shape)
        self.cam = orig_tpf.meta['CAMERA']
        self.ccd = orig_tpf.meta['CCD']
        self.sector = orig_tpf.meta['SECTOR']
        self.noise_func = noise_func
        self.noise_coeffs = noise_coeffs

        # define arrays that will get populated as the field is assembled
        self.field = np.zeros(self.orig_tpf.shape)
        self.bkg = np.zeros(self.orig_tpf.shape)
        self.sources = np.zeros(self.orig_tpf.shape)
        self.noise = np.zeros(self.orig_tpf.shape)

        # set the id number
        if id is not None:
            self.id = id 
        else:
            self.id = np.random.randint(100000000,999999999) # assign a random ID number

        # grab the prf
        self.prf = PRF.TESS_PRF(self.cam,self.ccd,self.sector,
                    self.orig_tpf.column,self.orig_tpf.row)
        
        # define the differential velocity aberrations
        # pos_corr should be in the format [x_diff, y_diff] in units of pixels
        if pos_time is not None:
            self.pos_corr = self.prep_positional_data(pos_time, pos_corr1, pos_corr2)
        else:
            # if no positional deviation information is supplied, populate the matrices with zeros
            self.pos_corr = np.zeros([2,len(self.orig_tpf)])

        # # fit a background and add it to the field
        # self.bkg = self.fit_bkg(polyorder=bkg_polyorder)
        
        # # final assembly
        # self.field = np.add(self.field, self.bkg)
        # add in noise from noise model here

    def prep_positional_data(self, pos_time, pos_corr1, pos_corr2):
        """Takes the pos_corr1 and pos_corr2 data and casts it into an array matching the time array, which can then be used to calculate the velocity aberrations."""
        # recast arrays to the right shape
        time_bin = self.orig_tpf.time.value
        pc1 = stats.binned_statistic(pos_time, pos_corr1, bins=time_bin)[0]
        pc2 = stats.binned_statistic(pos_time, pos_corr2, bins=time_bin)[0]
        # append extra element to the end to make the lengths match
        pc1 = np.append(pc1, pc1[-1])
        pc2 = np.append(pc2, pc2[-1])

        # forward fill any nans
        pc1 = trc.forward_fill(pc1)
        pc2 = trc.forward_fill(pc2)

        # after doing this, there may still be some nans left at the very beginning of the array
        # solve by backwards filling any nans as well
        pc1 = trc.backward_fill(pc1)
        pc2 = trc.backward_fill(pc2)

        return [pc1, pc2]

    def add_source(self, signal_func, buffer=3):
        """Adds a source with a given flux and position to the Field.
        signal - a 1d array that gives the behavior of the source over time, normalized to 1."""
        
        # convert source coords to pixel numbers and int pixel numbers
        pix1, pix2 = self.orig_tpf.wcs.all_world2pix(self.source_cat['ra'], self.source_cat['dec'], 0)
        pix1int = np.rint(pix1).astype(int)
        pix2int = np.rint(pix2).astype(int)

        # convert flux to mag (FIX LATER)
        flux_arr = trc.mag_to_flux(self.source_cat['Tmag'])

        # cut out indices where the target falls significantly outside the cutout
        cut = (pix1int < self.shape[0]+buffer) & (pix1int >= 0-buffer) & (pix2int < self.shape[1]+buffer) & (pix2int >= 0-buffer)
        source_cut = self.source_cat[cut]

        # retrieve the prf for this tpf
        # Suppose the following for a TPF of interest
        cam = self.orig_tpf.meta['CAMERA']
        ccd = self.orig_tpf.meta['CCD']
        sector = self.orig_tpf.meta['SECTOR']
        colnum = self.orig_tpf.column #middle of TPF?
        rownum = self.orig_tpf.row #middle of TPF?

        # add sources to the field, weighted by Tmag
        for source_ind in range(len(source_cut)):
        # for source_ind in range(1,2):
            # add the signal to the source
            # signal = flux_arr[source_ind] + 10*np.sin(self.orig_tpf.time.value * np.random.uniform(.5, 1.5))
            signal = flux_arr[source_ind] * signal_func(self.orig_tpf.time.value)

            # resample to make the prf for the source
            # source_prf = self.prf.locate(pix1[source_ind],pix2[source_ind], self.shape)
            try:
                source_prf = [self.prf.locate(pix1[source_ind]+self.pos_corr[0][i], pix2[source_ind]+self.pos_corr[1][i], self.shape) for i in range(len(self.orig_tpf))]
            except:
                print(source_ind)

            # apply the prf to the signl and add to image
            # gauss_blink = np.multiply(signal[:, np.newaxis, np.newaxis], source_prf[np.newaxis, :, :])
            gauss_blink = np.multiply(signal[:, np.newaxis, np.newaxis], source_prf)

            # add to the sources array
            self.sources = np.add(self.sources, gauss_blink)
            # field = np.add(field, gauss[np.newaxis, :, :])
            # field[:,pix1[cut],pix2[cut]] = 16 - self.source_cat['Tmag'][cut]

        pass

    def add_sources_from_catalog(self):
        """Given a catalog of sources, adds each of them to the Field."""
        return

    def calc_bkg(self, polyorder=3):
        """Adds background to the Field. Should only be called once."""
        self.bkg = trc.fit_bkg(self.orig_tpf, polyorder=polyorder)
        pass

    def set_noise_func(self, noise_func=None, noise_coeffs=None):
        """Allows the user to update the function and coefficients used to generate the empirical noise."""
        if noise_func is not None:
            self.noise_func = noise_func
        if noise_coeffs is not None:
            self.noise_coeffs = noise_coeffs
        pass

    def calc_empirical_noise(self):
        """Adds noise to the data based off the flux off pixel, using the noise function provided. Should only be called once the background and sources have been added."""
        logflux = np.log10(self.bkg + self.sources)
        self.noise = np.random.normal(0,self.noise_func(logflux,self.noise_coeffs))
        pass


    def assemble(self):
        """Once the background, sources, and noise have been added, assembles them all into the field."""
        self.field = self.bkg + self.sources + self.noise
        pass

    def to_tpf(self):
        """Converts the field to a TargetPixelFile."""
        self.assemble()
        out_file = copy.copy(self.orig_tpf)
        out_file = out_file * 0
        out_file += self.field
        return out_file
    