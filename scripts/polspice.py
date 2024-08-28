import os
import numpy as np
import healpy as hp
import fitsio
import matplotlib.pyplot as plt
import matplotlib as mpl
import camb
import transformcl
import wigner
from camb.sources import SplinedSourceWindow
from heracles.fields import Positions, Shears
from heracles.maps import Healpix
from heracles.maps import transform
from heracles.twopoint import angular_power_spectra, bin2pt
from heracles.fields import Visibility, Weights
from heracles.twopoint import mixing_matrices
from heracles.maps import map_catalogs
from heracles.core import TocDict

def cl2corr_mat(theta, lmax=None, spin1=0, spin2=0):
    if lmax is None:
        lmax = len(theta) - 1
    m = np.empty((len(theta), lmax + 1))
    f = (2 * np.arange(lmax + 1) + 1) / (4 * np.pi)
    for i, t in enumerate(theta):
        m[i] = f * wigner.wigner_dl(0, lmax, spin1, spin2, t)
    return m

def corr2cl_solve(mat, corr, spin1=0, spin2=0):
    k = max(abs(spin1), abs(spin2))
    cut_mat = mat[:, k:]
    return np.pad(np.linalg.solve(cut_mat, corr), (k, 0))

def corr2cl_lstsq(mat, corr):
    return np.linalg.lstsq(mat, corr, None)[0]

def apodize(x, x0=-2, k=50):
    return (1+np.exp(-k*(x-x0)))

# Resolution
nside = 1024
lmax = 1000
bin_num = 15

# Select Mask
mask_type = 'patch'

# Check if file already exists
pols_fname = "./samples/{}/polspice_data.npy".format(mask_type)
npols_fname = "./samples/{}/naive_polspice_data.npy".format(mask_type)
if os.path.isfile(pols_fname):
    pols_d_samples = np.load(pols_fname)
    npols_d_samples = np.load(npols_fname)
else:
    pols_d_samples = np.zeros((1, 3*bin_num))
    npols_d_samples = np.zeros((1, 3*bin_num))

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
mapper_mm = Healpix(2*nside, 2*lmax)
fields = {"P": Positions(None, mask="V"),
          "G": Shears(None, "RIGHT_ASCENSION", "DECLINATION", "G1", "-G2", "WEIGHT", mask="W"),
          "V": Visibility(mapper_mm),
          "W": Weights(mapper_mm, "RIGHT_ASCENSION", "DECLINATION", "WEIGHT")}
masks = {("V", 0): mask,
         ("W", 0): mask}
mask_alms = transform(fields, masks, progress=True)
mask_cls = angular_power_spectra(mask_alms)

# Apodize
Fm = transformcl.cltocorr(mask_cls[('V', 'V', 0, 0)][:1001])
Fm_full = transformcl.cltocorr(mask_cls[('V', 'V', 0, 0)])
Fm_apo = (Fm* apodize(np.log10(np.abs(Fm))))
Fm_apo_full = (Fm_full * apodize(np.log10(np.abs(Fm_full))))

# Polspice matrices
th = transformcl.theta(2*lmax+1)
mat_p = cl2corr_mat(th, lmax, 2, 2)
mat_m = cl2corr_mat(th, lmax, 2, -2)

i = len(pols_d_samples) - 1
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

    # Polspice
    cl_PP =  transformcl.corrtocl(transformcl.cltocorr(data_cls[('P', 'P', 0, 0)])/(Fm_apo))
    cl_EE_p_BB = data_cls[('G_E', 'G_E', 0, 0)] + data_cls[('G_B', 'G_B', 0, 0)]
    cl_EE_m_BB = data_cls[('G_E', 'G_E', 0, 0)] - data_cls[('G_B', 'G_B', 0, 0)]

    xi_m = mat_m @ cl_EE_m_BB
    xi_p = mat_p @ cl_EE_p_BB
    xi   = mat_m @ cl_EE_p_BB

    pols_Cl_BB = corr2cl_lstsq(mat_m, 0.5*(xi-xi_m)/Fm_apo_full)
    pols_Cl_EE = corr2cl_lstsq(mat_m, 0.5*(xi+xi_m)/Fm_apo_full)

    pols_cls = {}
    pols_cls[('G_B', 'G_B', 0, 0)] = pols_Cl_BB
    pols_cls[('G_E', 'G_E', 0, 0)] = pols_Cl_EE
    pols_cls[('P', 'P', 0, 0)]     = cl_PP

    # Naive Polspice
    cl_p = corr2cl_lstsq(mat_p, (1/Fm_apo_full) * (mat_p @ cl_EE_p_BB))
    cl_m = corr2cl_lstsq(mat_m, (1/Fm_apo_full) * (mat_m @ cl_EE_m_BB))
    npols_Cl_EE = 0.5*(cl_p + cl_m)
    npols_Cl_BB = 0.5*(cl_p - cl_m)

    npols_cls = {}
    npols_cls[('G_B', 'G_B', 0, 0)] = npols_Cl_BB
    npols_cls[('G_E', 'G_E', 0, 0)] = npols_Cl_EE
    npols_cls[('P', 'P', 0, 0)]     = cl_PP

    # To vector
    pols_d = np.array([])
    npols_d = np.array([])
    for cl_name in list(npols_cls.keys()):
        d_pols  = B @ pols_cls[cl_name]
        d_npols = B @ npols_cls[cl_name]
        pols_d = np.append(pols_d, d_pols)
        npols_d = np.append(npols_d, d_npols)

    # Append to samples
    pols_d_samples = np.vstack((pols_d_samples, pols_d))
    npols_d_samples = np.vstack((npols_d_samples, npols_d))

    # Save samples
    np.save(pols_fname, pols_d_samples[1:])
    np.save(npols_fname, npols_d_samples[1:])

    # Next sim
    i = i + 1
    sim_name = "./sims/sim_{}.fits".format(i)
