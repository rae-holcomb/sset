### the Field class
# Library of functions to be used when dealing with TRC data
import numpy as np
import lightkurve as lk
import math
import matplotlib.pyplot as plt
import copy
from scipy import stats
import warnings


import PRF

import astropy
from astroquery.mast import Catalogs
import astropy.units as u
import astropy.wcs as wcs
from astropy import table
# from astropy.stats import sigma_clip
# from scipy import ndimage
import typing

try:
    import sset.trc_funcs as trc
except: 
    import trc_funcs as trc

class Field():
    """Creates an array that 
    Inputs
        bkg_variability_generator - a FunctionSelector object that will generate variable timeseries for background stars
        noise_func - the function used to be for the empirical noise model
        noice_coeffs - the coefficients for the noise model
        buffer - this defines a buffer (in pixels) around the TPF cutout,stars that fall inside this buffer (even if they are not technically in the TPF) will be included in the field catalog
        offset_scale - the sigma of the distribution used to generate flux offsets for sources in the field"""
    def __init__(self, orig_tpf: lk.TessTargetPixelFile,
        source_catalog: astropy.table,
        bkg_polyorder: int, 
        bkg_variability_generator: typing.Callable=None,
        noise_func: typing.Callable=None, 
        noise_coeffs: np.array=None,
        pos_time=None, pos_corr1=None, pos_corr2=None,
        id: int=None,
        add_offset=False,
        offset_scale: float=0.0125,
        buffer: int=3
        ):

        # define some useful variables
        self.orig_tpf = orig_tpf
        self.time = orig_tpf.time.value
        self.source_catalog = source_catalog.copy()
        self.shape = orig_tpf.shape[1:]
        self.cam = orig_tpf.meta['CAMERA']
        self.ccd = orig_tpf.meta['CCD']
        self.sector = orig_tpf.meta['SECTOR']
        self.bkg_variability_generator = bkg_variability_generator
        self.noise_func = noise_func
        self.noise_coeffs = noise_coeffs
        self.buffer = buffer
        self.offset_scale = offset_scale

        # set the id number
        if id is not None:
            self.id = id 
        else:
            self.id = np.random.randint(100000000,999999999) # assign a random ID number

        # define a table that will track all the values used to make this field
        # self.sources = source_catalog.copy()['ID', 'ra', 'dec', 'pmRA', 'pmDEC', 'Tmag', 'GAIA', 'contratio', 'dstArcSec']
        # add columns for the Tmag in flux units (Tflux), pixel positions, variability function, params, and offset
        self.source_catalog['Tflux'] = trc.mag_to_flux(self.source_catalog['Tmag'])
        # pixel position columns
        pix1, pix2 = self.orig_tpf.wcs.all_world2pix(self.source_catalog['ra'], self.source_catalog['dec'], 0)
        self.source_catalog['pix1'] = pix1
        self.source_catalog['pix2'] = pix2
        self.source_catalog['pix1int'] = np.rint(pix1).astype(int)
        self.source_catalog['pix2int'] = np.rint(pix2).astype(int)
        # variability columns
        self.source_catalog['signal_function'] = None
        self.source_catalog['signal_params'] = {}


        # cut out sources from the catalog that fall significantly outside the TPF
        # NOTE: This means that self.source_catalog may be smaller than the original input catalog!
        cut = (self.source_catalog['pix1'] < self.shape[0]+self.buffer) & (self.source_catalog['pix1'] >= 0-self.buffer) & (self.source_catalog['pix2'] < self.shape[1]+self.buffer) & (self.source_catalog['pix2'] >= 0-self.buffer)
        # self.sources = self.sources[cut]
        self.source_catalog = self.source_catalog[cut]

        # define arrays that will get populated as the field is assembled
        self.field = np.zeros(self.orig_tpf.shape)
        self.bkg = np.zeros(self.orig_tpf.shape)
        self.signals1D = np.ones((len(self.source_catalog),len(self.orig_tpf.time)))
        self.signals2D = np.zeros(self.orig_tpf.shape)
        self.noise = np.zeros(self.orig_tpf.shape)

        # flags for turning certain things on or off
        self.add_offset = add_offset

        # calculate offsets for the sources if requested
        # Note: add keyword option to change the sigma on the offset function
        if self.add_offset:
            self.source_catalog['offset'] = self.generate_offset(size=len(self.source_catalog))
        else:
            self.source_catalog['offset'] = 1.

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

        # fit a background and add it to the field
        self.bkg = trc.fit_bkg(self.orig_tpf, polyorder=bkg_polyorder)

        # Apply a noise model. May want to rerun this after adding in sources.
        
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

    def add_premade_1D_signal(self, idx: int, premade_signal: np.array, params: typing.Dict={}, signal_func=None, **kwargs):
        """
        Use to add a signal to a star if you have already generated the flux array, which must be same length as the time array.

        idx - the index corresponding to the stellar source you are injecting a signal into
        signal_func - the TSGenerator object that can be used to recreate this signal
        params - the dictionary of parameters to be passed to the signal_func to recreate this signal
        """
        if len(premade_signal) != len(self.time):
            warnings.warn('Provided signal is not the same length as the time array.')
            return

        # update the class variables to reflect the injected signal
        self.signals1D[idx,:] = premade_signal
        self.source_catalog['signal_function'][idx] = signal_func
        self.source_catalog['signal_params'][idx] = params
        pass

    def add_1D_signal(self, idx: int, random_signal=True, signal_func=None, **kwargs):
        """NEW VERSION
        Adds a source with a given flux and position to the signals1D array.

        signal - a 1d array that gives the behavior of the source over time, normalized to 1. Note that if random_signal=False then a signal_func MUST be provided.
        
        Inputs:
            idx - the index (or array of indices) in the source catalog of the source
            signal_func - (callable) the function to be used to generate the signal
            random_signal - (bool) if true, will use the FunctionSelector to generate a random signal
            """
        if type(idx) == int:
            idx = np.array([idx])

        # add sources to the field, weighted by Tmag
        for source_ind in idx:
            row = self.source_catalog[source_ind]
            
            # generate signal
            if random_signal:
                # if requested, use the FunctionSelector to generate a random signal
                selected_function, signal, params = self.bkg_variability_generator.instantiate_function(self.orig_tpf.time.value)
                # record what function was used
                self.source_catalog['signal_function'][idx] = selected_function
            else:
                # otherwise, use the provided TSGenerator and params
                signal, params = signal_func.generate_signal(self.orig_tpf.time.value)
                # record what function was used
                self.source_catalog['signal_function'][idx] = signal_func.name
                # print('not a random signal')

            # update the class variables to reflect the inject signal
            self.signals1D[source_ind,:] = signal
            self.source_catalog['signal_params'][idx] = params
        pass

    def convert_to_2D(self, idx: int):
        """Helper function that takes a 1D timeseries and smears it out into 2D using the appropriate PRF, then returns the 2D array. Does NOT modify the state of the Field object at all.

        Inputs:
            idx - (int) the index of the source to be smeared.
        """
        row = self.source_catalog[idx]

        # scale by the flux and apply offset
        signal_scaled = row['Tflux'] * self.signals1D[idx,:] * row['offset']

        # resample to make the prf for the source
        # source_prf = self.prf.locate(pix1[idx],pix2[idx], self.shape)
        try:
            source_prf = [self.prf.locate(row['pix1']+self.pos_corr[0][i], row['pix2']+self.pos_corr[1][i], self.shape) for i in range(len(self.orig_tpf))]
        except:
            print("Couldn't find PRF for target index " + str(idx))

        # apply the prf to the signl and add to image
        signal2D = np.multiply(signal_scaled[:, np.newaxis, np.newaxis], source_prf)

        return signal2D

    def convert_field_to_2D(self):
        """
        Once all desired 1D signals have been generated, call this function to use the PRF to smear them into 2D.
        """
        # clear the signals2D matrix
        self.signals2D = np.zeros(self.orig_tpf.shape)

        for source_ind in range(len(self.source_catalog)):
            signal2D = self.convert_to_2D(source_ind)
            self.signals2D = np.add(self.signals2D, signal2D)
        pass

    def add_signal_legacy(self, signal_func, buffer=3):
        """OLD VERSION, DELETE ONCE YOU HAVE THE NEW ONE WORKING
        Adds a source with a given flux and position to the Field.
        signal - a 1d array that gives the behavior of the source over time, normalized to 1."""
        
        # # convert source coords to pixel numbers and int pixel numbers
        # pix1, pix2 = self.orig_tpf.wcs.all_world2pix(self.source_catalog['ra'], self.source_catalog['dec'], 0)
        # pix1int = np.rint(pix1).astype(int)
        # pix2int = np.rint(pix2).astype(int)

        # # convert flux to mag (FIX LATER)
        # flux_arr = trc.mag_to_flux(self.source_catalog['Tmag'])

        # # cut out indices where the target falls significantly outside the cutout
        # cut = (pix1int < self.shape[0]+buffer) & (pix1int >= 0-buffer) & (pix2int < self.shape[1]+buffer) & (pix2int >= 0-buffer)
        # source_cut = self.source_catalog[cut]

        # # retrieve the prf for this tpf
        # # Suppose the following for a TPF of interest
        # cam = self.orig_tpf.meta['CAMERA']
        # ccd = self.orig_tpf.meta['CCD']
        # sector = self.orig_tpf.meta['SECTOR']
        # colnum = self.orig_tpf.column #middle of TPF?
        # rownum = self.orig_tpf.row #middle of TPF?

        # add sources to the field, weighted by Tmag
        for source_ind in range(len(self.source_catalog)):
        # for source_ind in range(1,2):
            # add the signal to the source
            signal = flux_arr[source_ind] * signal_func(self.orig_tpf.time.value)

            # add an offset if requested
            # NOTE: NEED TO MAKE NOTE OF THIS SOMEWHERE IN THE FIELD OBJECT
            if self.add_offset == True:
                offset = self.generate_offset()
                signal *= offset

            # resample to make the prf for the source
            # source_prf = self.prf.locate(pix1[source_ind],pix2[source_ind], self.shape)
            try:
                source_prf = [self.prf.locate(pix1[source_ind]+self.pos_corr[0][i], pix2[source_ind]+self.pos_corr[1][i], self.shape) for i in range(len(self.orig_tpf))]
            except:
                print(source_ind)

            # apply the prf to the signl and add to image
            # gauss_blink = np.multiply(signal[:, np.newaxis, np.newaxis], source_prf[np.newaxis, :, :])
            gauss_blink = np.multiply(signal[:, np.newaxis, np.newaxis], source_prf)

            # add to the signals2D array
            self.signals2D = np.add(self.signals2D, gauss_blink)
            # field = np.add(field, gauss[np.newaxis, :, :])
            # field[:,pix1[cut],pix2[cut]] = 16 - self.source_catalog['Tmag'][cut]

        pass

    def add_signals_from_catalog(self):
        """Given a catalog of sources, adds each of them to the Field."""
        return

    def calc_bkg(self, polyorder=3):
        """Calculates the background. Calling this will overwrite the existing background."""
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
        logflux = np.log10(self.bkg + self.signals2D)
        self.noise = np.random.normal(0,self.noise_func(logflux,self.noise_coeffs))
        pass

    def generate_offset(self, offset_scale=None, size=None) -> float:
        """Generates a multiplicative offset for each source in the field. Currently, just using a gaussian centered on 1 with a sigma of .0125, but maybe make this more complicated later to reflect empirical offsets for each sector.
        offset_scale - lets you overwrite the class-defined offset_scale"""
        if offset_scale is not None:
            return np.random.normal(loc=1., scale=offset_scale, size=size)
        else:
            return np.random.normal(loc=1., scale=self.offset_scale, size=size)

    def assemble(self):
        """Once the background, sources, and noise have been added, assembles them all into the field."""
        self.convert_field_to_2D()
        self.calc_empirical_noise()
        self.field = self.bkg + self.signals2D + self.noise
        pass

    def to_tpf(self):
        """Converts the field to a TargetPixelFile."""
        self.assemble()
        out_file = copy.copy(self.orig_tpf)
        out_file = out_file * 0
        out_file += self.field
        return out_file
    
    def nemesis_lc_extraction(self, threshold, cadence,
            verbose=True,
            use_SPOC_aperture=True,
            use_centroid=False,
            use_circ=False,
            use_sources_in_aperture=False):

        tpf = self.to_tpf()
        sap_lc = trc.nemesis_SAP(tpf,threshold,cadence,
                    verbose=verbose,
                    use_SPOC_aperture=use_SPOC_aperture,
                    use_centroid=use_centroid,
                    use_circ=use_circ,
                    use_sources_in_aperture=use_sources_in_aperture)
        return sap_lc


class MultiSectorField:
    """Takes in multiple field objects and populates them with continuous variability functions.

    field_arr - an array of Field objects that has been initialized but no further operations made to it.
    source_catalog - 


    tpf_arr - array of lightkurve TargetPixelFile objects. Only populated after calling self.to_tpf()
    catalog_arr - super catalog containing all targets that appear in any sector
    time - an array with all timestamps from all sectors
    """
    def __init__(self, 
                field_arr: typing.List[Field],
                source_catalog: astropy.table,
                bkg_variability_generator: typing.Callable=None,
                ):
        self.field_arr = field_arr
        self.source_catalog = source_catalog
        self.bkg_variability_generator = bkg_variability_generator

        # # define useful variables
        self.tpf_arr = [None] * len(self.field_arr) # will only be populated after self.to_tpf() is called
        self.time = np.concatenate([fd.time for fd in field_arr])
        self.signals1D = np.ones((len(self.source_catalog),len(self.time)))

        # add columns to the source catalog to keep track of the functions we're using to generate the variability
        self.source_catalog['signal_function'] = None
        self.source_catalog['signal_params'] = {}


        # # create a super catalog that contains all targets from the source catalogs of all the sectors
        # self.catalog_arr = table.unique(table.vstack([fd.source_catalog for fd in field_arr]), keys='ID')
        
        # background & offset is automatically populated for each field(I think), but remember to add in noise at the end!


        return

    def add_premade_1D_signal(self, idx: int, premade_signal: np.array, params: typing.Dict={}, signal_func=None, **kwargs):
        """
        Use to add a signal to a star if you have already generated the flux array, which must be same length as the time array.

        idx - the index corresponding to the stellar source you are injecting a signal into
        signal_func - the TSGenerator object that can be used to recreate this signal
        params - the dictionary of parameters to be passed to the signal_func to recreate this signal
        """
        if len(premade_signal) != len(self.time):
            warnings.warn('Provided signal is not the same length as the time array.')
            return

        # update the class variables to reflect the injected signal
        self.signals1D[idx,:] = premade_signal
        self.source_catalog['signal_function'][idx] = signal_func
        self.source_catalog['signal_params'][idx] = params

        # also update each individual field with the provided signal
        for fd in self.field_arr:
            # grab just the portion of the time series that shows up in that field
            premade_signal_cut = premade_signal[(self.time >= min(fd.time)) & (self.time <= max(fd.time))]

            # add and update params
            fd.add_premade_1D_signal(idx, premade_signal_cut, params=params, signal_func=signal_func)
        return

    def add_1D_signal(self, idx: int, random_signal=True, signal_func=None, **kwargs):
        """
        Adds a source with a given flux and position to the signals1D array.

        signal - a 1d array that gives the behavior of the source over time, normalized to 1. Note that if random_signal=False then a signal_func MUST be provided.
        
        Inputs:
            idx - the index (or array of indices) in the source catalog of the source
            signal_func - (callable) the function to be used to generate the signal
            random_signal - (bool) if true, will use the FunctionSelector to generate a random signal
            """
        if type(idx) == int:
            idx = np.array([idx])

        # add sources to the field, weighted by Tmag
        for source_ind in idx:
            row = self.source_catalog[source_ind]
            
            # generate signal
            if random_signal:
                # if requested, use the FunctionSelector to generate a random signal
                selected_function, signal, params = self.bkg_variability_generator.instantiate_function(self.time)
                # record what function was used
                self.source_catalog['signal_function'][idx] = selected_function
            else:
                # otherwise, use the provided TSGenerator and params
                signal, params = signal_func.generate_signal(self.time)
                # record what function was used
                self.source_catalog['signal_function'][idx] = signal_func.name
                # print('not a random signal')

            # update the class variables to reflect the inject signal
            self.signals1D[source_ind,:] = signal
            self.source_catalog['signal_params'][idx] = params

            # also update each individual field with the provided signal
            for fd in self.field_arr:
                # grab just the portion of the time series that shows up in that field
                signal_cut = signal[(self.time >= min(fd.time)) & (self.time <= max(fd.time))]

                # add and update params
                fd.add_premade_1D_signal(idx, signal_cut, params=params, signal_func=signal_func)
        pass


    def add_signal(self, targe_id: int, random_signal=True, signal_func=None, **kwargs):
        """Generates a signal across the timestamps for all sectors, then adds them to the field."""
        return
    
    def add_noise(self):
        """Calls functions to calculate the empirical noise and offset for each field."""
        for fd in self.field_arr :
            fd.calc_empirical_noise()

    def to_tpfs(self):
        for ind in range(len(self.tpf_arr)):
            self.tpf_arr[ind] = self.field_arr[ind].to_tpf()

        return self.tpf_arr
