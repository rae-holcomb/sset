# Library of functions to be used when dealing with TRC data
import numpy as np
import lightkurve as lk
import math
import matplotlib.pyplot as plt
import copy
import typing 

import PRF


from astroquery.mast import Catalogs
import astropy.units as u
import astropy.wcs as wcs
from scipy import stats
from astropy.stats import sigma_clip
import astropy.table
from scipy import ndimage
from scipy.optimize import curve_fit

try:
    import sset.field as Field
except:
    import field as Field

# helper functions

def forward_fill(arr):
    """Backward fills nan values in an array."""
    out = arr.copy()
    for col_idx in range(1, len(out)):
        if np.isnan(out[col_idx]):
            out[col_idx] = out[col_idx - 1]
    return out

def backward_fill(arr):
    """Forward fills nan values in an array."""
    out = arr.copy()
    for col_idx in range(len(out)-2, -1, -1):
        if np.isnan(out[col_idx]):
            out[col_idx] = out[col_idx + 1]
    return out

def roundup(x, pow=0):
    """
    Rounds up to the nearest power of ten, given by pow.
    """
    return int(math.ceil(x / (10 ** (pow)))) * (10 ** (pow))


def pixels_to_radius(cutout_size, pow=-2, round=True):
    """
    Given the side length of a cut out in pixels, converts it to the radius of a cone search in degrees, rounded up to the nearest `pow` power of 10. Use round to turn on/off the rounding up feature.
    """
    if round:
        return roundup(cutout_size / np.sqrt(2) * 21 / 3600, pow=pow)
    else:
        return cutout_size / np.sqrt(2) * 21 / 3600


def pix_to_arcsec(pix):
    return pix * 21.0


def pix_to_degrees(pix):
    return pix * 21.0 / 3600


def flux_to_mag(flux, reference_mag=20.44):
    """Converts a TESS magnitude to a flux in e-/s. The TESS reference magnitude is taken to be 20.44. If needed, the Kepler reference flux is 1.74e5 electrons/sec.
    
    Parameters
    ----------
    flux : float
        The total flux of the target on the CCD in electrons/sec.
    reference_mag: int
        The zeropoint reference magnitude for TESS. Typically 20.44 +/-0.05.
    reference_mag: float

    Returns
    -------
    Tmag: float
        TESS magnitude of the target.
    
    """
    # kepler_mag = 12 - 2.5 * np.log10(flux / reference_flux)
    mag = -2.5 * np.log10(flux) + reference_mag
    return mag


def mag_to_flux(Tmag, reference_mag=20.44):
    """Converts a TESS magnitude to a flux in e-/s. The TESS reference magnitude is taken to be 20.44. If needed, the Kepler reference flux is 1.74e5 electrons/sec.
    
    Parameters
    ----------
    Tmag: float
        TESS magnitude of the target.
    reference_mag: int
        The zeropoint reference magnitude for TESS. Typically 20.44 +/-0.05.

    Returns
    -------
    flux : float
        The total flux of the target on the CCD in electrons/sec.
    """
    # fkep = (10.0 ** (-0.4 * (mag - 12.0))) * 
    return 10 ** (-(Tmag - reference_mag)/2.5)

def mag_to_flux_fake(mag):
    """A fake function for converting magnitudes to flux, just for development purposes. Don't use for anything dimmer than 16th mag!"""
    return (16 - mag)*100


def makeGaussian(size_x, size_y, fwhm=1, center=None):
    """Make a square gaussian kernel.

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
    arr = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm**2)
    return arr

def convert_to_distribution(input: typing.Union[float, stats.rv_continuous]) -> stats.rv_continuous:
    """Converts a float to a stats.rv_continuous object."""
    # logic to recast float to dist if necessary
    if isinstance(input, (float, int)):
        return stats.uniform(loc=input, scale=0)
    else:
        return input

def convert_to_incl(y):
    """Takes a number in [0,1] and transforms it to an inclination according to y = sin^2(i). Useful for calculating distributions that are uniform in sin^2(i) space."""
    return np.arcsin(np.sqrt(y))

def exponential(x, coeffs):
    """
    Defines a parabola. Intended to be used by curvefit function. coeffs takes the form [a, b, c] for the exponential form y = a * np.exp(b * x) + c.
    """
    return coeffs[0] * np.exp(coeffs[1] * x) + coeffs[2]

def polynomial(x,coeffs):
    """
    Defines a parabola. Intended to be used by curvefit function..
    """
    y = np.zeros_like(x)
    
    for ind, val in enumerate(coeffs):
        y += val * np.power(x, len(coeffs) - ind - 1)
    return y

# Goddard work


def set_seed():
    """Set seed for all processes."""
    raise NotImplementedError
    # return


def get_cutout(target, cutout_size=20, sector=None):
    """Given RA/DEC and size (in pix by pix), grabs cut out of real TESS data.

    Basically just calls the search_tesscut() function of lightcurve. (and honestly probably just replace it with that later."""
    tpf_cutout = lk.search_tesscut(target, sector=sector).download(
        cutout_size=cutout_size
    )
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
    catalog_data = Catalogs.query_object(
        target, catalog="TIC", radius=pixels_to_radius(cutout_size)
    )
    # apply brightness cutoff
    catalog = catalog_data[catalog_data["Tmag"] < Tmag_cutoff]
    return catalog

def convert_array(arr, position, new_shape, new_position=None):
    """"Converts the size of a 2D boolean mask, given absolute positions on some larger grid of the old mask and the new mask.
    
    Parameters
    ----------
    arr : ndarray
        2D boolean mask.
    position : tuple
        Absolute position of the original array on some larger grid.
    new_shape : tuple
        Shape of the new boolean mask
    new_position : tuple
        Absolute position of the new array on some larger grid.

    Returns
    -------
    result : ndarray
        Converted 2D boolean mask.
    """
    # extract positions and sizes for easy reference
    a1, b1 = len(arr), len(arr[0])
    x1, y1 = position
    a2, b2 = new_shape
    
    if new_position is None:
        x2, y2 = x1 - a1//2, y1 - b1//2
    else:
        x2, y2 = new_position
    
    # relative positions
    dx = x1 - x2
    dy = y1 - y2

    # set the default values to false
    larger_array = [[False] * b2 for _ in range(a2)]
    
    # loop through and update
    for i in range(a1):
        for j in range(b1):
            new_i = dx + i
            new_j = dy + j
            if 0 <= new_i < a2 and 0 <= new_j < b2:
                larger_array[new_i][new_j] = arr[i][j]
    
    return np.array(larger_array)

def convert_aperture_mask(pipeline_tpf, larger_tpf):
    """"Takes the pipeline aperture mask from a TPF and converts it to fit a larger sized TPF cutout.
    
    Parameters
    ----------
    pipeline_tpf : lk.TessTargetPixelFile
        A target pixel file which has a pipeline aperture mask.
    larger_tpf : lk.TessTargetPixelFile
        A larger targetpixel file (such as one obtained from TESScut) which you want to apply the aperture mask to.
    
    Returns
    -------
    result : ndarray
        Converted 2D boolean mask.
    """
    arr = pipeline_tpf.pipeline_mask
    position = (pipeline_tpf.row, pipeline_tpf.column)
    new_shape = larger_tpf.pipeline_mask.shape
    new_position = (larger_tpf.row, larger_tpf.column)
    return convert_array(arr, position, new_shape, new_position)

def estimate_bkg_OLD(tpf_cutout, source_cat):
    """Given a real TESS cut out and associated sources, masks the sources and fits a low order polynomial to estimate a model for the background. Should return an array the same shape as the input tpf_cutous"""
    # currently just returning the median flux value of each cadence as a placeholder
    # christina said that she'd give me code to do this properly on Tuesday
    med = np.median(tpf_cutout.flux.value, axis=[1, 2])
    return np.multiply(
        np.ones_like(tpf_cutout.flux.value), med[:, np.newaxis, np.newaxis]
    )


def fit_bkg(tpf: lk.TessTargetPixelFile, polyorder: int = 1) -> np.ndarray:
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
    # Notes for understanding this function
    # All arrays in this func will have dimensions drawn from one of the following: [ntimes, ncols, nrows, npix, ncomp]
    #   ntimes = number of cadences
    #   ncols, nrows = shape of tpf
    #   npix = ncols*nrows, is the length of the unraveled vectors
    #   ncomp = num of components in the polynomial

    # Error catching
    if not isinstance(tpf, lk.TessTargetPixelFile):
        raise ValueError("Input a TESS Target Pixel File")

    if (np.product(tpf.shape[1:]) < 100) | np.any(np.asarray(tpf.shape[1:]) < 6):
        raise ValueError("TPF too small. Use a bigger cut out.")

    # Grid for calculating polynomial
    R, C = np.mgrid[: tpf.shape[1], : tpf.shape[2]].astype(float)
    R -= tpf.shape[1] / 2
    C -= tpf.shape[2] / 2

    # nested b/c we run twice, once on each orbit
    def func(tpf):
        # Design matrix
        A = np.vstack(
            [
                R.ravel() ** idx * C.ravel() ** jdx
                for idx in range(polyorder + 1)
                for jdx in range(polyorder + 1)
            ]
        ).T

        # Median star image
        m = np.median(tpf.flux.value, axis=0)
        # Remove background from median star image
        mask = ~sigma_clip(m, sigma=3).mask.ravel()
        # plt.imshow(mask.reshape(m.shape))
        bkg0 = A.dot(
            np.linalg.solve(A[mask].T.dot(A[mask]), A[mask].T.dot(m.ravel()[mask]))
        ).reshape(m.shape)

        # m is the median frame
        m -= bkg0

        # Include in design matrix
        A = np.hstack([A, m.ravel()[:, None]])

        # Fit model to data, including a model for the stars in the last column
        f = np.vstack(tpf.flux.value.transpose([1, 2, 0]))
        ws = np.linalg.solve(A.T.dot(A), A.T.dot(f))
        # shape of ws is (num of times, num of components)
        # A . ws gives shape (npix, ntimes)

        # Build a model that is just the polynomial
        model = (
            (A[:, :-1].dot(ws[:-1]))
            .reshape((tpf.shape[1], tpf.shape[2], tpf.shape[0]))
            .transpose([2, 0, 1])
        )
        # model += bkg0
        return model

    # Break point for TESS orbit
    # currently selects where the biggest gap in cadences is
    # could cause problems in certain cases with lots of quality masking! Think about how to handle bit masking
    b = np.where(np.diff(tpf.cadenceno) == np.diff(tpf.cadenceno).max())[0][0] + 1

    # Calculate the model for each orbit, then join them
    model = np.vstack([func(tpf) for tpf in [tpf[:b], tpf[b:]]])

    return model


def measure_noise(tpf, medfil_kernel=25, window=25, mask=None):
    """Measure the characteristic noise of a pixel as a function of mean flux. medfil_kernel is the kernel size used when calculating the median filter. Window is in units of data points and will be affected by what the cadence of your data is, controls how long of a window will be used to calculate the standard deviation."""
    # pseudo code
    # # get median filter of raw data
    # mf_raw = ndimage.median_filter(tpf.flux.value, size=[medfil_kernel,1,1])


    # subtract background, median filter that, subtract that
    bkg = fit_bkg(tpf, polyorder=3)
    mf1 = ndimage.median_filter(tpf.flux.value - bkg, size=[medfil_kernel,1,1])
    cleaned = tpf.flux.value - bkg - mf1

    # calculate running median and std 
    time = tpf.time.value
    time_bin = np.arange(time[0], time[-1], .5)
    std_bin = stats.binned_statistic(time, cleaned.reshape([tpf.shape[0],tpf.shape[1]*tpf.shape[2]]).T, bins=time_bin, statistic='std')[0]
    med_bin = stats.binned_statistic(time, tpf.flux.value.reshape([tpf.shape[0],tpf.shape[1]*tpf.shape[2]]).T, bins=time_bin, statistic='median')[0]

    # save datapoint as function of [median of raw data, std]
    return med_bin.flatten(), std_bin.flatten()


def generate_point_field(tpf_cutout, source_cat, plot=False):
    """Given a source catalog and tpf_cutout, generates a field with point sources. Currently is very jank and just proof of concept. The list of sources in read in and then the single pixel nearest their location is given a value weighed by 16-Tmag. The plot keyword will plot the tpf_cutout and the basic field, rotated so that they hopefully line up.

    Known issues:
    - Values are not real fluxes
    - the pixel numbers on the axes don't line up with the real tpf pixel numbers
    - jank rounding for ra/dec --> pixel conversion
    """
    # set up field
    field = np.ones(tpf_cutout.shape[1:])

    # convert source coords to integer pixel numbers
    pix1, pix2 = np.rint(
        tpf_cutout.wcs.all_world2pix(source_cat["ra"], source_cat["dec"], 0)
    ).astype(int)

    # cut out indices where the target falls outside the cutout
    shape = tpf_cutout.shape[1:]
    cut = (pix1 < shape[1]) & (pix1 >= 0) & (pix2 < shape[2]) & (pix2 >= 0)

    # add sources to the field, weighted by Tmag
    field[:, pix1[cut], pix2[cut]] = 16 - source_cat["Tmag"][cut]

    # plot if requested
    if plot:
        plt.imshow(np.rot90(field[0]))
        plt.show()
        tpf_cutout.plot()

    return field

def toy_diff_velocity_aberration(time, amp1, amp2):
    """Calculates a toy model of the differential velocity abberation in pixels on the CCD over a period of time. Orbits are currently assumed to be 14 days, and start at the beginning of the time stamp. This may be modified later to be more realistic.
    
    Returns the position drift in units of pixels as [x_diff, y_diff]. Inputs are the time array and the amplitude of the drift along the x and y directions."""
    orbit_per = 14
    x_diff = amp1 * np.sin( np.pi / (orbit_per) * (time - time[0]) )**2
    y_diff = amp2 * np.sin( np.pi / (orbit_per) * (time - time[0]) )**2
    return x_diff, y_diff


def generate_signal(time):
    """Swappable rotation/other signal-generating code. Should always return [time, flux]"""
    return np.sin(time) + 1


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
    """Add reasonable noise to the background pixels."""
    return


def generate_error_ext():
    """Possibly add later, applies the relationship of flux to flux error to make error extensions if needed."""
    return


####################################
### LC EXTRACTION FUNCS FROM DAX ###
####################################

# Frankenstein'd code from NEMESIS pipeline (https://iopscience.iop.org/article/10.3847/1538-3881/abedb3/meta)

def create_circular_mask(median_image, reference_pixel, radius=1):
    h, w = median_image.shape[:2]
    if reference_pixel is None: # use the middle of the image, this is a fair assumption for TPFs which are pre-centered,
        #                       # on targets. FFIs, not so much... 
        reference_pixel = (int(h/2), int(w/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(reference_pixel[0], reference_pixel[1], h-reference_pixel[0], w-reference_pixel[1])
    #
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - reference_pixel[0])**2 + (Y-reference_pixel[1])**2)
    #
    mask = dist_from_center <= radius
    return mask
#
def thresholdmask(hdu,reference_pixel,threshold,use_centroid=False,use_circ=True):
    import numpy as np
    ### threshold higher is more conservative, lower is more liberal
    ### define image from image data
    median_image = np.nanmedian(hdu[1].data['FLUX'], axis=0)
    vals=median_image[np.isfinite(median_image)].flatten()
    ###
    from astropy.stats.funcs import median_absolute_deviation as MAD
    ###
    # A value for the number of sigma by which a pixel needs to be
    # brighter than the median flux to be included in the aperture:
    ###
    MADcut = ( 1.4826*MAD(vals)*threshold +np.nanmedian(median_image))
    threshold_mask = np.nan_to_num(median_image) > MADcut
    #
    #
    if (reference_pixel == None):
        # return all regions above threshold
        return threshold_mask
    #
    #
    ###
    from scipy.ndimage import label #<--converts image to 1s and 0s as "labels"
    labels = label(threshold_mask)[0]
    label_args = np.argwhere(labels > 0)
    ###
    ### For all pixels above threshold, compute distance to reference pixel:
    distances = [np.hypot(crd[0], crd[1]) for crd in label_args - np.array([reference_pixel[1], reference_pixel[0]])]
    ###
    ### Which label corresponds to the closest pixel?
    closest_arg = label_args[np.argmin(distances)]
    closest_label = labels[closest_arg[0], closest_arg[1]]
    ###
    threshmask=labels==closest_label
    ###
    # BELOW SHOULD ONLY BE DONE IF use_centroid==True!!!
    #if use_centroid==True:
    #check if thresmask is ACTUALLY close to reference_pixel 
    ###
    tx,ty=centroid_quadratic(median_image, threshmask, reference_pixel)
    ###
    dist_from_ref = np.round(np.sqrt( (tx-reference_pixel[0])**2 + (ty-reference_pixel[1])**2  ),2)
    if use_circ==True:
        if dist_from_ref > 1.5:
            r=1.5
            print('selected aperture mask is ',dist_from_ref,'pixels from reference pixel! Using circular aperture (r = '+str(r)+' pixels) around reference pixel instead')
            #
            #
            circ_mask = create_circular_mask(median_image, reference_pixel, radius=r)
            threshmask = circ_mask
        else:
            threshmask=threshmask
    if use_circ==False:
        threshmask=threshmask
    ###
    if use_centroid==False:
        reference_pixel = reference_pixel
    if use_centroid==True:
        reference_pixel = centroid_quadratic( median_image, threshmask, reference_pixel)
    return threshmask            
#
def centroid_quadratic(data, mask, reference_pixel):
    # Step 1: identify the patch of 3x3 pixels (z_)
    # that is centered on the brightest pixel (xx, yy)
    if mask is not None:
        data = data * mask
    arg_data_max = np.nanargmax(data)
    yy, xx = np.unravel_index(arg_data_max, data.shape)
    #    
    # Make sure the 3x3 patch does not leave the image bounds
    if yy < 1:
        yy = 1
    if xx < 1:
        xx = 1
    if yy > (data.shape[0] - 2):
        yy = data.shape[0] - 2
    if xx > (data.shape[1] - 2):
        xx = data.shape[1] - 2
    #
    z_ = data[yy-1:yy+2, xx-1:xx+2]
    #
    # Next, we will fit the coefficients of the bivariate quadratic with the
    # help of a design matrix (A) as defined by Eqn 20 in Vakili & Hogg
    # (arxiv:1610.05873). The design matrix contains a
    # column of ones followed by pixel coordinates: x, y, x**2, xy, y**2.
    #
    A = np.array([[1, -1, -1, 1,  1, 1],
                  [1,  0, -1, 0,  0, 1],
                  [1,  1, -1, 1, -1, 1],
                  [1, -1,  0, 1,  0, 0],
                  [1,  0,  0, 0,  1, 0],
                  [1,  1,  0, 1,  0, 0],
                  [1, -1,  1, 1, -1, 1],
                  [1,  0,  1, 0,  0, 1],
                  [1,  1,  1, 1,  1, 1]])
    #
    # We also pre-compute $(A^t A)^-1 A^t$, cf. Eqn 21 in Vakili & Hogg.
    At = A.transpose()
    #
    # In Python 3 this can become `Aprime = np.linalg.inv(At @ A) @ At`
    Aprime = np.matmul(np.linalg.inv(np.matmul(At, A)), At)
    #
    # Step 2: fit the polynomial $P = a + bx + cy + dx^2 + exy + fy^2$
    # following Equation 21 in Vakili & Hogg.
    # In Python 3 this can become `Aprime @ z_.flatten()`
    a, b, c, d, e, f = np.matmul(Aprime, z_.flatten()) #dot product
    #
    # Step 3: analytically find the function maximum,
    # following https://en.wikipedia.org/wiki/Quadratic_function
    det = 4.0 * d * f - (e ** 2.0)
    #     if abs(det) < 1e-6:
    #         return np.nan, np.nan  # No solution
    xm = - (2.0 * f * b - c * e) / det
    ym = - (2.0 * d * c - b * e) / det
    #
    return np.array(xx + xm), np.array(yy + ym)


def nemesis_SAP(tpf,threshold,cadence,
        verbose=True,
        use_SPOC_aperture=False,
        use_centroid=False,use_circ=True,
        use_sources_in_aperture=False):
    """Adapted from code provided by Dax Feliz, which was adapted from the NEMESIS pipeline."""
    # pull out useful variables
    hdu = tpf.hdu
    quality_mask = hdu[1].data['QUALITY']!=0
    fmt = tpf.time.format 
    scale = tpf.time.scale
    flux_unit = tpf.flux.unit
    mission = tpf.mission
    cutoutsize = max(tpf.shape[1:])

    # get the reference pixel
    if (cadence=='short') or (cadence=='fast') or (cadence=='2 minute') or (cadence=='20 second') :
        x=hdu[1].header['1CRPX4']-1 # in this case, TPFs are already centered
        y=hdu[1].header['2CRPX4']-1 # subtracting by one places coord on center, from testing they're always off by 1...
    if (cadence=='long') or (cadence=='30 minute') or (cadence=='10 minute'):
        x=hdu[1].header['1CRPX4']-0.5 #subtracting by 0.5 places coord on center of pixel
        y=hdu[1].header['2CRPX4']-0.5 #otherwise it's on corner of pixel
    reference_pixel=[x,y]

    median_image = np.nanmedian(hdu[1].data['FLUX'][~quality_mask], axis=0)
    ###
    ###
    flux = hdu[1].data['FLUX'][~quality_mask]
    rawtime = hdu[1].data['TIME'][~quality_mask]
    #include only finite values, excluding NaNs
    m = np.any(np.isfinite(flux),axis=(1,2))
    rawtime = np.ascontiguousarray(rawtime[m],dtype=np.float64)
    flux = np.ascontiguousarray(flux[m],dtype=np.float64)
    ###
    ###
    # find dimmest pixels using inverted threshold mask
    bkg_mask = ~thresholdmask(hdu,reference_pixel=None,threshold=1/1000,\
                              use_centroid=use_centroid,use_circ=use_circ)
    ###
    ### double check that reference pixel makes sense
    if (reference_pixel[0] > cutoutsize) or (reference_pixel[1] > cutoutsize):
        print('refernece pixel is larger than cutout!?!')
        print(reference_pixel)
        reference_pixel = [cutoutsize/2,cutoutsize/2] #assume it's in middle
        print(reference_pixel)
    ###
    ###
    # select brightest pixels using threshold mask
    try:
        T=threshold
        threshmask=thresholdmask(hdu,reference_pixel,threshold=T,\
                                 use_centroid=use_centroid,use_circ=use_circ)
    except ValueError as e:
        try:
            T=threshold/2.0
            print('threshold too high, no pixels selected. Trying half of input threshold: ',str(T))
            threshmask=thresholdmask(hdu,reference_pixel,threshold=T,\
                                     use_centroid=use_centroid,use_circ=use_circ)
        except ValueError as ee:
            try:
                T=threshold/3.0
                print('threshold STILL too high, no pixels selected. Trying third of input threshold: ',str(T))
                threshmask=thresholdmask(hdu,reference_pixel,threshold=T,\
                                         use_centroid=use_centroid,use_circ=use_circ)
            except ValueError as eee:
                print('setting threshold=1 (last resort)')
                T=1.0
                try:
                    threshmask=thresholdmask(hdu,reference_pixel,threshold=T,\
                                             use_centroid=use_centroid,use_circ=use_circ)
                except ValueError as eeee:
                    print('Ok, I tried...')
                    pass
    try:
        pix_mask=threshmask
        if verbose==True:
            print('selected threshold: ',T)
    except UnboundLocalError as UE:
        print('unable to find suitable threshold mask')
        return
    #
    if (cadence=='short') or (cadence=='2 minute') or (cadence=='20 second') or (cadence=='fast'):
        if use_SPOC_aperture==True:
            print('using SPOC aperture')
            try:
                pipeline_mask = hdu[2].data & 2 > 0
            except TypeError:  # Early versions of TESScut returned floats in HDU 2
                pipeline_mask = np.ones(hdu[2].data.shape, dtype=bool)
            pix_mask = pipeline_mask
            bkg_mask = ~pipeline_mask
        if use_SPOC_aperture==False:
            if verbose==True:
                print('using threshold mask')
    ###
    ###
    rawflux = np.nansum(flux[:, pix_mask], axis=-1)
    bkgFlux = np.nansum(flux[:, bkg_mask], axis=-1)
    #
    Npixbkg = len(np.where(bkg_mask == True)[0])
    Npixaper= len(np.where(pix_mask == True)[0])
    
    bkgFlux = bkgFlux/Npixbkg #normalize background
    
    rawsap_flux = rawflux - (bkgFlux * Npixaper)
    sap_flux = rawsap_flux #/ np.nanmedian(rawsap_flux)
    
    nanmask = np.where(np.isfinite(sap_flux)==True)[0]
    #print('len SAP nanmask',len(nanmask))
    error = np.abs( sap_flux / np.nanmedian(np.nansum(sap_flux)/np.nanmedian(sap_flux)))
    error = np.abs( sap_flux / np.nanmedian(np.nansum(sap_flux)))    
    #print('len E before nanmask',len(error))
    error =error[nanmask]
    #
    if verbose==True:
        print('len check ',' T', len(rawtime),' SAP',len(sap_flux), ' E ', len(error))
        print('shape check ',' T', np.shape(rawtime),' SAP',np.shape(sap_flux), ' E ', np.shape(error))
    #SAP_LC = pd.DataFrame({"Time":rawtime,"RAW SAP Flux":rawsap_flux,"SAP Flux":sap_flux,"SAP Error":error, "Background Flux":bkgFlux})
    import astropy.units as u
    from astropy.time import Time
    rawtime = Time(rawtime, format=fmt, scale=scale)
    # sap_flux *= (u.def_unit('e-')/u.second)
    sap_flux *= flux_unit
    error = u.Quantity(error, sap_flux.unit)
    print(sap_flux.unit,error.unit)
    SAP_LC = lk.LightCurve(time=rawtime,flux=sap_flux,
                           flux_err=error)
    SAP_LC.mission=mission
    if verbose==True:
        print('RAW len check:', len(rawtime),len(sap_flux),len(error))
    return SAP_LC



##### OLD

# variability functions


def calc_r_var(lc_norm):
    """Calculates the 5th to 95th percentile variability. LCs should be normalized before calling this function."""
    n5 = np.percentile(lc_norm.flux.value, 5)
    n95 = np.percentile(lc_norm.flux.value, 95)
    return np.abs(n95 - n5)


def peak_periodogram_strength(lc):
    """Calculates the power of the highest peak in the LS periodogram. LCs should be normalized before calling this function."""
    return lc.to_periodogram().max_power


# other helper functions
def num_continuous(arr, allow_gap=0):
    """Returns the length of the longest sequence of consecutive integers in an ordered list.
    Setting allow_gap indicates how big of an integer gap can be ignored when counting continuous sequences."""
    if len(arr) == 1:
        return arr

    count = 1
    max_count = 1

    ind_start, ind_end = 0, 1
    max_ind_start, max_ind_end = 0, 1

    for i in range(len(arr) - 1):
        if arr[i + 1] - arr[i] <= 1 + allow_gap:
            count += 1
            ind_end += 1

            if count > max_count:
                max_count = count
                max_ind_start = ind_start
                max_ind_end = ind_end
        else:
            count = 1
            ind_start = i + 1
            ind_end = i + 2

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
    x = np.arange(-3 * sigma, 3 * sigma)
    # note that we divide by .997 to preserve the normalization and make the
    # area under the truncated gaussian equal to 1
    return (
        1.0
        / 0.997
        * 1.0
        / (np.sqrt(2.0 * np.pi) * sigma)
        * np.exp(-((x / sigma) ** 2.0) / 2.0)
    )


def gaussian_smooth(y, fwhm):
    """fwhm is in units of data points, convert to real time units (days or s) later."""
    return np.convolve(y, gaussian(fwhm), mode="same")
