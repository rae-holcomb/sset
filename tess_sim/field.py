### the Field class
# Library of functions to be used when dealing with TRC data
import numpy as np
import lightkurve as lk
import math
import matplotlib.pyplot as plt
import copy

import PRF

from astroquery.mast import Catalogs
import astropy.units as u
import astropy.wcs as wcs
# from astropy.stats import sigma_clip
# from scipy import ndimage

import trc_funcs as trc