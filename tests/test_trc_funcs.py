# import sys
# sys.path.append('../sset/')

import numpy as np
import lightkurve as lk
import math
import sset.trc_funcs as trc
import sset.field as Field
import sset.generator_classes as gen
from scipy import stats

# FIX THESE AND MAKE THEM REAL TESTS LATER

def pixels_to_radius_test():
    """
    Test the conversion of pixels to arcseconds. 
    """
    px = 20
    trc.roundup(px / np.sqrt(2) * 21 / 3600, pow=-2)
    assert trc.pixels_to_radius(20) == 0.9

def roundup_test():
    """
    Test roundup function works correctly. 
    """
    assert trc.roundup(40.11, pow=-1) == 40.2
    assert trc.roundup(40.11, pow=0) == 41.

def fit_bkg_test():
    """
    
    """
    bkg = trc.estimate_bkg(tpf_cutout, source_cat)
    # print(np.median(tpf_cutout[0].flux.value))
    # np.median(tpf_cutout.flux.value, axis=[1,2])

    # bkg = trc.estimate_bkg(tpf_cutout, source_cat)
    # print(np.shape(bkg))
    # plt.imshow(bkg[0])

    # b = bkg[:, np.newaxis, np.newaxis]
    # b[0,:,:]
    # # b, _ = np.broadcast_arrays(b, a)
    # # b
    assert True# that the shape of the bkg == shape of the input cut out
    assert True# that the median value of the bkg model and the image are within 10% of each other

def generate_basic_field_test():
    # test that the shape of the output field matches the input tpf_cutout
    # some index handling testing?
    # test a funky shape of cut out
    # some flux level testing?
    assert True

def flux_mag_converstion_test():
    """Test that converting from mag to flux and back keeps things consistent."""
    tmag1 = 10
    assert(tmag1 == trc.flux_to_mag(trc.mag_to_flux(10)))

def convert_to_distribution_test():
    # check that the types come out right
    assert(isinstance(convert_to_distribution_test(10), stats.rv_continuous))
    assert(isinstance(convert_to_distribution_test(12.3), stats.rv_continuous))
    assert(isinstance(convert_to_distribution_test(stats.norm(loc=1, scale=3)), stats.rv_continuous))

    # check that the numbers come out right
    a = -1.2
    b = 300
    assert(trc.convert_to_distribution(a).rvs() == a)
    assert(trc.convert_to_distribution(b).rvs() == b)

def convert_aperture_mask():
    # test to make sure the output has the mask in the right place
    # add test for non-square TPFs or mismatched shapes
    # also test convert_array
    raise NotImplementedError


