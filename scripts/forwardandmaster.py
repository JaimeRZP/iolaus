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
bin_num = 15

# Select Mask
mask_type = 'patch'

# Check if file already exists
forward_data_fname = "./samples/{}/data.npy".format(mask_type)
forward_theory_fname = "./samples/{}/forward_theory.npy".format(mask_type)
master_data_fname = "./samples/{}/master_data.npy".format(mask_type)
master_theory_fname = "./samples/{}/master_theory.npy".format(mask_type)
if os.path.isfile(forward_data_fname):
    f_d_samples = np.load(forward_data_fname)
    f_t_samples = np.load(forward_theory_fname)
    m_d_samples = np.load(master_data_fname)
    m_t_samples = np.load(master_theory_fname)
else:
    f_d_samples = np.zeros((1, 3*bin_num))
    f_t_samples = np.zeros((1, 3*bin_num))
    m_d_samples = np.zeros((1, 3*bin_num))
    m_t_samples = np.zeros((1, 3*bin_num))

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

# bin mixing matrix
mms_qq = {}
for keys in list(mms.keys()):
    mms_qq[keys] = B @ mms[keys] @ B2.T

# Invert binned Mixing Matrix
inv_mms_qq = {}
for keys in list(mms_qq.keys()):
    inv_mms_qq[keys] = np.linalg.inv(mms_qq[keys])

# Forward Model
forward_cls = {}

forward_cls[('G_B', 'G_B', 0, 0)] = mms[('G_E', 'G_E', 0, 0)] @ theory_cls[('G_B', 'G_B', 0, 0)] + mms[('G_B', 'G_B', 0, 0)] @ theory_cls[('G_E', 'G_E', 0, 0)]
forward_cls[('G_E', 'G_E', 0, 0)] = mms[('G_E', 'G_E', 0, 0)] @ theory_cls[('G_E', 'G_E', 0, 0)] + mms[('G_B', 'G_B', 0, 0)] @ theory_cls[('G_B', 'G_B', 0, 0)]
forward_cls[('P', 'P', 0, 0)]     = mms[('P', 'P', 0, 0)] @ theory_cls[('P', 'P', 0, 0)]

i = len(f_d_samples) - 1
sim_name = "./sims/sim_{}.fits".format(i)
while os.path.isfile(sim_name):
    print(i)
    # Load sim
    theory_map = hp.read_map(sim_name, field=[0, 1, 2])

    # Alms
    mapper = Healpix(nside, lmax)
    fields = {
        "P": Positions(mapper, "RIGHT_ASCENSION", "DECLINATION", mask="V"),
        "G": Shears(mapper, "RIGHT_ASCENSION", "DECLINATION", "G1", "-G2",
                    "WEIGHT", mask="W")}

    maps = {('P', 0): theory_map[0]*mask,
            ('G', 0): np.array([theory_map[1]*mask, theory_map[2]*mask], np.dtype(float, metadata={'spin': 2})),}
    alms = transform(fields, maps, progress=False)

    # Two-point Statistics
    data_cls = angular_power_spectra(alms)

    # To vector
    forward_t = np.array([])
    forward_d = np.array([])
    master_t = np.array([])
    master_d = np.array([])
    for cl_name in list(forward_cls.keys()):
        f_d = B @ data_cls[cl_name]
        f_t = B @ forward_cls[cl_name]
        m_d = inv_mms_qq[cl_name] @ f_d
        m_t = inv_mms_qq[cl_name] @ f_t
        forward_d = np.append(forward_d, f_d)
        forward_t = np.append(forward_t, f_t)
        master_d = np.append(master_d, m_d)
        master_t = np.append(master_t, m_t)

    # Append to samples
    f_d_samples = np.vstack((f_d_samples, forward_d))
    f_t_samples = np.vstack((f_t_samples, forward_t))
    m_d_samples = np.vstack((m_d_samples, master_d))
    m_t_samples = np.vstack((m_t_samples, master_t))

    # Save samples
    np.save(forward_data_fname,   f_d_samples[1:])
    np.save(forward_theory_fname, f_t_samples[1:])
    np.save(master_data_fname,    m_d_samples[1:])
    np.save(master_theory_fname,  m_t_samples[1:])

    # Next sim
    i = i + 1
    sim_name = "./sims/sim_{}.fits".format(i)
