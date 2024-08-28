import os
import numpy as np
import healpy as hp
import fitsio
import matplotlib.pyplot as plt
import matplotlib as mpl

import camb
from camb.sources import SplinedSourceWindow

from heracles.fields import Positions, Shears
from heracles.maps import Healpix
from heracles.maps import transform
from heracles.twopoint import angular_power_spectra, bin2pt
from heracles.fields import Visibility, Weights
from heracles.twopoint import mixing_matrices
from heracles.maps import map_catalogs
from heracles.core import TocDict

# Resolution
nside = 1024
lmax = 1000

# Make Mask
mask_type = "euclid_south"
theory_map = hp.read_map("../scripts/sims/sim_0.fits", field=[0,1,2])
mask = np.ones_like(theory_map[1])
pixel_theta, pixel_phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

if mask_type == 'patch':
    mask[np.pi/3 > pixel_theta] = 0.0
    mask[pixel_theta > 2*np.pi/3] = 0.0
    mask[pixel_phi > np.pi/2] = 0.0
    mask[np.pi/8> pixel_phi] = 0.0

if mask_type == 'one_third':
    mask[np.pi/3 > pixel_theta] = 0.0

if mask_type == 'half':
        mask[np.pi/2 > pixel_theta] = 0.0

if mask_type == 'two_thirds':
        mask[2*np.pi/3 > pixel_theta] = 0.0

if mask_type == 'euclid_north':
        vmap_n = hp.read_map('/home/jaimerz/Documents/UCL/heracles/data/vmap_wlfs2_dr1n_vis24.5_nomag.fits')
        vmap_n[vmap_n == hp.UNSEEN] = 0.
        vmap_n = hp.ud_grade(vmap_n, nside)
        mask *= vmap_n

if mask_type == 'euclid_south':
        vmap_s = hp.read_map('/home/jaimerz/Documents/UCL/heracles/data/vmap_wlfs2_dr1s_vis24.5_nomag.fits')
        vmap_s[vmap_s == hp.UNSEEN] = 0.
        vmap_s = hp.ud_grade(vmap_s, nside)
        mask *= vmap_s

if mask_type == 'euclid':
        vmap_n = hp.read_map('/home/jaimerz/Documents/UCL/heracles/data/vmap_wlfs2_dr1n_vis24.5_nomag.fits')
        vmap_n[vmap_n == hp.UNSEEN] = 0.
        vmap_n = hp.ud_grade(vmap_n, nside)
        vmap_s = hp.read_map('/home/jaimerz/Documents/UCL/heracles/data/vmap_wlfs2_dr1s_vis24.5_nomag.fits')
        vmap_s[vmap_s == hp.UNSEEN] = 0.
        vmap_s = hp.ud_grade(vmap_s, nside)
        vmap = vmap_n + vmap_s
        mask *= vmap

# Mixing matrices
# Mixing Matrix
mapper_mm = Healpix(2*nside, 2*lmax)
fields = {"P": Positions(None, mask="V"),
          "G": Shears(None, "RIGHT_ASCENSION", "DECLINATION", "G1", "-G2", "WEIGHT", mask="W"),
          "V": Visibility(mapper_mm),
          "W": Weights(mapper_mm, "RIGHT_ASCENSION", "DECLINATION", "WEIGHT")}
masks = {("V", 0): mask,
         ("W", 0): mask}
mask_alms = transform(fields, masks, progress=False)
mask_cls = angular_power_spectra(mask_alms)
mms = mixing_matrices(fields, mask_cls,
    l3max=2*lmax+lmax, l2max=2*lmax, l1max=lmax, progress=True)

# Save
fname = "./masks/{}.fits".format(mask_type)
hp.fitsfunc.write_map(fname, mask, overwrite=True)

fname = "./masks/{}.npy".format(mask_type)
dict_mms = {}
for mms_key in list(mms.keys()):
    dict_mms[mms_key] = mms[mms_key] 
print(type(dict_mms))
np.save(fname, dict_mms)
