# Library of functions to be used when dealing with TRC data
import numpy as np
import lightkurve as lk
import math
import matplotlib.pyplot as plt

from astroquery.mast import Catalogs
import astropy.units as u
import astropy.wcs as wcs
from astropy.stats import sigma_clip
# helper functions

def roundup(x, pow=0):
    """Rounds up to the nearest power of ten, given by pow."""
    return int(math.ceil(x / (10**(pow)))) * (10**(pow))

def pixels_to_radius(cutout_size, pow=-2, round=True):
    """Given the side length of a cut out in pixels, converts it to the radius of a cone search in degrees, rounded up to the nearest `pow` power of 10. Use round to turn on/off the rounding up feature."""
    if round:
        return roundup(cutout_size / np.sqrt(2) * 21 / 3600, pow=pow)
    else:
        return cutout_size / np.sqrt(2) * 21 / 3600

def pix_to_arcsec(pix):
    return(pix*21.)

def pix_to_degrees(pix):
    return(pix*21./3600)

def flux_to_mag(flux, reference_flux=1.74e5):
    """NOTE: ref mag is a place holder from kepler right now!"""
    kepmag = 12 - 2.5 * np.log10(flux/reference_flux)
    return kepmag

def mag_to_flux(mag, reference_flux=1.74e5):
    """NOTE: ref mag is a place holder from kepler right now!"""
    fkep = (10.0**(-0.4*(mag - 12.0)))*reference_flux
    # f12 = 1.74e5 # electrons/sec
    return fkep

def makeGaussian(size_x, size_y, fwhm = 1, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum (units of pixels), which
    can be thought of as an effective radius.

    Note: NOT normalized correctly! But that shouldn't matter since this is a place holder for now.
    """
    x = np.arange(0, size_x, 1, float)[np.newaxis, :]
    y = np.arange(0, size_y, 1, float)[:, np.newaxis]

    if center is None:
        x0 = size_x // 2
        y0 = size_y // 2
    else:
        x0 = center[0]
        y0 = center[1]

    # make and normalize so that the who image sums to 1
    arr = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return arr

# Goddard work

def set_seed():
    """Set seed for all processes."""
    return

def get_cutout(target, cutout_size=20, sector=None):
    """Given RA/DEC and size (in pix by pix), grabs cut out of real TESS data. 
    
    Basically just calls the search_tesscut() function of lightcurve. (and honestly probably just replace it with that later."""
    tpf_cutout = lk.search_tesscut(target, sector=sector).download(cutout_size=cutout_size)
    return tpf_cutout

def get_catalog(target, cutout_size=20, Tmag_cutoff=16):
    """Given RA/DEC and size (in pix by pix), grabs catalog of sources brighter than 16th mag within a radius from the TIC. Radius should be sqrt(2*(sidelength)^2. Returns an astropy table.
    Basically uses astroquery.mast.query_catalog
    
    Inputs:
        target - simbad resolvable name or string of ra/dec
        cutout_size - size of tpf cutout in pixels
        Tmag_cutoff - dimmest Tmag to include in catalog
    Outputs:
        catalog - astropy Table object with sources from the TIC    
    """
    # query catalog
    catalog_data = Catalogs.query_object(target, catalog="TIC", radius=pixels_to_radius(cutout_size)) 
    # apply brightness cutoff
    catalog = catalog_data[catalog_data['Tmag'] < Tmag_cutoff]
    return catalog

def estimate_bkg_OLD(tpf_cutout, source_cat):
    """Given a real TESS cut out and associated sources, masks the sources and fits a low order polynomial to estimate a model for the background. Should return an array the same shape as the input tpf_cutous"""
    # currently just returning the median flux value of each cadence as a placeholder
    # christina said that she'd give me code to do this properly on Tuesday
    med = np.median(tpf_cutout.flux.value, axis=[1,2])
    return np.multiply(np.ones_like(tpf_cutout.flux.value), med[:, np.newaxis, np.newaxis])

def fit_bkg(tpf:lk.TessTargetPixelFile, polyorder:int=1) -> np.ndarray:
    """Fit a simple 2d polynomial background to a TPF
    
    Parameters
    ----------
    tpf: lightkurve.TessTargetPixelFile
        Target pixel file object
    polyorder: int
        Polynomial order for the model fit.
        
    Returns
    -------
    model : np.ndarray
        Model for background with same shape as tpf.shape
    """
    
    if not isinstance(tpf, lk.TessTargetPixelFile):
        raise ValueError("Input a TESS Target Pixel File")
    
    if (np.product(tpf.shape[1:]) < 100) | np.any(np.asarray(tpf.shape[1:]) < 6):
        raise ValueError("TPF too small. Use a bigger cut out.")
        
        
    # Grid for calculating polynomial
    R, C = np.mgrid[:tpf.shape[1], :tpf.shape[2]].astype(float)
    R -= tpf.shape[1]/2
    C -= tpf.shape[2]/2
    
    
    def func(tpf):
        # Design matrix
        A = np.vstack([R.ravel()**idx*C.ravel()**jdx for idx in range(polyorder + 1) for jdx in range(polyorder + 1)]).T
        
        # Median star image
        m = np.median(tpf.flux.value, axis=0)
        # Remove background from median star image
        mask = ~sigma_clip(m, sigma=3).mask.ravel()
        #plt.imshow(mask.reshape(m.shape))
        bkg0 = A.dot(np.linalg.solve(A[mask].T.dot(A[mask]), A[mask].T.dot(m.ravel()[mask]))).reshape(m.shape)
        
        m -= bkg0

        # Include in design matrix
        A = np.hstack([A, m.ravel()[:, None]])
        
        # Fit model to data, including a model for the stars
        f = np.vstack(tpf.flux.value.transpose([1, 2, 0]))
        ws = np.linalg.solve(A.T.dot(A), A.T.dot(f))
        
        # Build a model that is just the polynomial
        model = (A[:, :-1].dot(ws[:-1])).reshape((tpf.shape[1], tpf.shape[2], tpf.shape[0])).transpose([2, 0, 1])
        model += bkg0
        return model
    
    # Break point for TESS orbit
    b = np.where(np.diff(tpf.cadenceno) == np.diff(tpf.cadenceno).max())[0][0] + 1
    
    # Calculate the model for each orbit, then join them
    model = np.vstack([func(tpf) for tpf in [tpf[:b], tpf[b:]]])
    return model

def generate_basic_field(tpf_cutout, source_cat, plot=False):
    """Given a source catalog and tpf_cutout, generates a field with point sources. Currently is very jank and just proof of concept. The list of sources in read in and then the single pixel nearest their location is given a value weighed by 16-Tmag. The plot keyword will plot the tpf_cutout and the basic field, rotated so that they hopefully line up. 

    Known issues: 
    - Values are not real fluxes 
    - the pixel numbers on the axes don't line up with the real tpf pixel numbers 
    - jank rounding for ra/dec --> pixel conversion
    """
    # set up field
    field = np.zeros_like(tpf_cutout.flux.value)

    # convert source coords to integer pixel numbers
    pix1, pix2 = np.rint(tpf_cutout.wcs.all_world2pix(source_cat['ra'], source_cat['dec'], 0)).astype(int)

    # cut out indices where the target falls outside the cutout
    shape = np.shape(field)
    cut = (pix1 < shape[1]) & (pix1 >= 0) & (pix2 < shape[2]) & (pix2 >= 0)

    # add sources to the field, weighted by Tmag
    field[:,pix1[cut],pix2[cut]] = 16 - source_cat['Tmag'][cut]

    # plot if requested
    if plot:
        plt.imshow(np.rot90(field[0]))
        plt.show()
        tpf_cutout.plot();
    
    return field


def generate_signal(time):
    """Swappable rotation/other signal-generating code. Should always return [time, flux]"""
    return np.sin(time)

def get_prf():
    """Gets prf for a particular source"""
    return 

def stack_prfs():
    """Does the matrix stuff to get the prfs for all sources in the image"""
    return 

def add_source_noise():
    """Uses empirical TESS relationship to add gaussian noise pixel-by-pixel to sources."""
    return

def add_bkg_noise():
    """Add reasonable noise to the backgroun pixels."""
    return 

def generate_error_ext():
    """Possibly add later, applies the relationship of flux to flux error to make error extensions if needed."""
    return 








##### OLD

# variability functions

def calc_r_var(lc_norm):
    """Calculates the 5th to 95th percentile variability. LCs should be normalized before calling this function."""
    n5 = np.percentile(lc_norm.flux.value, 5)
    n95 = np.percentile(lc_norm.flux.value, 95)
    return(np.abs(n95-n5))

def peak_periodogram_strength(lc):
    """Calculates the power of the highest peak in the LS periodogram. LCs should be normalized before calling this function."""
    return lc.to_periodogram().max_power

# other helper functions
def num_continuous(arr, allow_gap=0):
    """Returns the length of the longest sequence of consecutive integers in an ordered list.
    Setting allow_gap indicates how big of an integer gap can be ignored when counting continuous sequences."""
    if(len(arr)==1):
        return arr
    
    count = 1
    max_count = 1
    
    ind_start, ind_end = 0, 1
    max_ind_start, max_ind_end= 0, 1

    for i in range(len(arr)-1):
        if arr[i+1] - arr[i] <= 1 + allow_gap :
            count += 1
            ind_end += 1

            if count > max_count :
                max_count = count
                max_ind_start = ind_start
                max_ind_end = ind_end
        else:
            count = 1
            ind_start = i+1
            ind_end = i+2
    
    return arr[max_ind_start:max_ind_end]

def gaussian(fwhm):
    """
    Creates a gaussian with a given FWHM as preparation for convolution with a light curve.
    Args:
        fwhm (:obj:`float`): the full width half max of the desired gaussian.
        bs (:obj:`float`): the size of the bins, in units of seconds.
    Returns:
        gaussian (:obj:`arr`): a gaussian.
    """
    sigma = fwhm / 2.355
    x = np.arange(-3*sigma, 3*sigma)
    # note that we divide by .997 to preserve the normalization and make the
    # area under the truncated gaussian equal to 1
    return 1./.997 * 1./(np.sqrt(2.*np.pi) * sigma) * np.exp(-(x/sigma)**2./2.)

def gaussian_smooth(y, fwhm):
    """fwhm is in units of data points, convert to real time units (days or s) later."""
    return np.convolve(y, gaussian(fwhm), mode="same")


