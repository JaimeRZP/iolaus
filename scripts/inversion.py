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
lmax = 1500
bin_num = 15

# Select Mask
mask_type = 'patch'

# Check if file already exists
data_fname = "./samples/{}/inversion_data.npy".format(mask_type)
theory_fname = "./samples/{}/theory.npy".format(mask_type)
if os.path.isfile(data_fname):
    d_samples = np.load(data_fname)
    t_samples = np.load(theory_fname)
else:
    d_samples = np.zeros((1, 3*bin_num))
    t_samples = np.zeros((1, 3*bin_num))

# Load cls
theory_cls = np.load("./sims/theory.npy", allow_pickle=True).item()

# Binning
#bin_edges = np.logspace(np.log10(10), np.log10(lmax+1), bin_num+1).astype(int)
bin_edges = np.linspace(10, lmax+1, bin_num+1).astype(int)
B = np.diag(np.ones(lmax+1))
B = bin2pt(B, bin_edges, 'B')
binned_ls = B['L']
B = B['B']

n, m = B.shape
B2 = np.zeros((n, 2*lmax+1))
B2[:, :lmax+1] = B

# Apply Mask
mask_path = "./masks/{}.fits".format(mask_type)
mask = hp.read_map(mask_path)

# Mixing Matrix
mms_path = "./masks/{}.npy".format(mask_type)
mms = np.load(mms_path, allow_pickle=True).item()

# Invert Mixing Matrix
inv_mms = {}
M_PP = mms[('P', 'P', 0, 0)]
M_EE = np.hstack([mms[('G_E', 'G_E', 0, 0)], mms[('G_E', 'G_B', 0, 0)]])
M_BB = np.hstack([mms[('G_E', 'G_B', 0, 0)], mms[('G_B', 'G_B', 0, 0)]])
M_EE_BB = np.vstack([M_EE, M_BB])
inv_mms['M_EE_BB']        = np.linalg.pinv(M_EE_BB)
inv_mms[('P', 'P', 0, 0)] = np.linalg.pinv(M_PP)

i = len(d_samples)-1
sim_name = "./sims/sim_{}.fits".format(i)
while os.path.isfile(sim_name):
    print(i)
    # Load sim
    theory_map = hp.read_map(sim_name, field=[0,1,2])

    # Alms
    mapper = Healpix(nside, lmax)
    fields = {
        "P": Positions(mapper, "RIGHT_ASCENSION", "DECLINATION", mask="V"),
        "G": Shears(mapper, "RIGHT_ASCENSION", "DECLINATION", "G1", "-G2", "WEIGHT", mask="W"),
    }

    maps = {('P', 0): theory_map[0]*mask,
            ('G', 0): np.array([theory_map[1]*mask, theory_map[2]*mask],
                                np.dtype(float, metadata={'spin': 2}))}
    alms = transform(fields, maps, progress=False)

    # Two-point Statistics
    data_cls = angular_power_spectra(alms)

    # Inverse Model
    inversion_cls = {}
    cl_EE_BB = np.append(data_cls[('G_E', 'G_E', 0, 0)], data_cls[('G_B', 'G_B', 0, 0)])
    cl_EE_BB = inv_mms['M_EE_BB'] @ cl_EE_BB
    inversion_cls[('G_B', 'G_B', 0, 0)] = cl_EE_BB[:2*lmax+1]
    inversion_cls[('G_E', 'G_E', 0, 0)] = cl_EE_BB[2*lmax+1:]
    inversion_cls[('P', 'P', 0, 0)]     = inv_mms[('P', 'P', 0, 0)] @ data_cls[('P', 'P', 0, 0)]

    # To vector
    inversion_t = np.array([])
    inversion_d = np.array([])
    for cl_name in list(inversion_cls.keys()):
        d = B2 @ inversion_cls[cl_name]
        t = B2 @ theory_cls[cl_name]
        inversion_t = np.append(inversion_t, t)
        inversion_d = np.append(inversion_d, d)

    # Append to samples
    d_samples = np.vstack((d_samples, inversion_d))
    t_samples = np.vstack((t_samples, inversion_t))

    # Save samples
    np.save(data_fname, d_samples[1:])
    np.save(theory_fname, t_samples[1:])

    # Next sim
    i = i + 1
    sim_name = "./sims/sim_{}.fits".format(i)
