import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import heracles
import numpy as np
import healpy as hp
import camb
import camb.correlations
from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights
from camb.correlations import gauss_legendre_correlation as glc
from scipy.interpolate import interp1d

def bin2pt(arr, bins, name):
    """Compute binned two-point data."""

    def norm(a, b):
        """divide a by b if a is nonzero"""
        out = np.zeros(np.broadcast(a, b).shape)
        return np.divide(a, b, where=(a != 0), out=out)

    # flatten list of bins
    bins = np.reshape(bins, -1)
    m = bins.size

    # shape of the data
    n, *ds = np.shape(arr)
    ell = np.arange(n)

    # create the structured output array
    # if input data is multi-dimensional, then so will the `name` column be
    binned = np.empty(m - 1, [(name, float, ds) if ds else (name, float)])

    # get the bin index for each ell
    i = np.digitize(ell, bins)
    assert i.size == ell.size
    wb = np.bincount(i, minlength=m)[1:m]
    # bin data in ell
    for j in np.ndindex(*ds):
        x = (slice(None), *j)
        binned[name][x] = norm(np.bincount(i, arr[x], m)[1:m], wb)

    # all done
    return binned

# Resolution
nside = 1024
lmax = 1500
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
bin_edges = np.logspace(np.log10(10), np.log10(lmax+1), bin_num+1)
B = np.diag(np.ones(lmax+1))
B = bin2pt(B, bin_edges, 'B')
B = B['B']

# Binned ls
l = np.arange(lmax+1)
prefac = l * (l + 1) / (2 * np.pi)
q = B @ l

# Apply Mask
mask_path = "./masks/{}.fits".format(mask_type)
mask = hp.read_map(mask_path)
mapper_mm = HealpixMapper(2*nside, 2*lmax)
fields = {"P": Positions(None, mask="V"),
          "G": Shears(None, "RIGHT_ASCENSION", "DECLINATION", "G1", "-G2", "WEIGHT", mask="W"),
          "V": Visibility(mapper_mm),
          "W": Weights(mapper_mm, "RIGHT_ASCENSION", "DECLINATION", "WEIGHT")}
masks = {("V", 0): mask,
         ("W", 0): mask}
mask_alms = heracles.transform(fields, masks, progress=True)
mask_cls = heracles.angular_power_spectra(mask_alms)

# Apodize
mask_for_corr = np.transpose([
  mask_cls * prefac,
  np.zeros(lmax+1),
  np.zeros(lmax+1),
  np.zeros(lmax+1)])
mask_corr, mask_corr_x, mask_corr_w = glc(mask_for_corr,
                                          lmax=lmax,
                                          sampling_factor=1)
mask_corr[:, 0] += mask_cls[0]/(4*np.pi)
xi_mask = mask_corr.T[0]
xi_mask *= logistic(np.log10(abs(xi_mask)))

i = len(pols_d_samples) - 1
sim_name = "./sims/sim_{}.fits".format(i)
while os.path.isfile(sim_name):
    print(i)
    # Load sim
    theory_map = hp.read_map(sim_name, field=[0, 1, 2])

    # Alms
    mapper = HealpixMapper(nside, lmax)
    fields = {
        "P": Positions(mapper, "RIGHT_ASCENSION", "DECLINATION", mask="V"),
        "G": Shears(mapper, "RIGHT_ASCENSION", "DECLINATION", "G1", "-G2",
                    "WEIGHT", mask="W")}

    maps = {('P', 0): theory_map[0]*mask,
            ('G', 0): np.array([theory_map[1]*mask, theory_map[2]*mask], np.dtype(float, metadata={'spin': 2})),}
    alms = heracles.transform(fields, maps, progress=False)

    # Two-point Statistics
    data_cls = heracles.angular_power_spectra(alms)

    # Polspice
    cls_for_corr = np.array([
        data_cls['TT'],
        data_cls['EE'],
        data_cls['BB'],
        np.zeros(lmax+1)])
    cls_for_corr *= prefac
    data_corr, corr_x, corr_w = glc(cls_for_corr.T,
                                    lmax=lmax,
                                    sampling_factor=1)
    data_corr[:, 0] += data_cls['TT'][0]/(4*np.pi)
    xi_TT = data_corr.T[0]
    xi_p  = data_corr.T[1]
    xi_m  = data_corr.T[2]

    # Eq90 plus
    xi_eq90_plus = Eq90_plus(corr_x, xi_p)

    # Eq90 minus
    xi_eq90_minus = Eq90_minus(corr_x, xi_m)

    # Transform back to Cl
    pols_plus_corrs = np.array([
        xi_TT/xi_mask,
        xi_p/xi_mask,
        xi_eq90_plus/xi_mask,
        np.zeros(lmax+1)])
    pols_plus_cls_list = camb.correlations.corr2cl(pols_plus_corrs.T,
                                            corr_x, corr_w, lmax)
    prefac[0] = 1
    pols_plus_cls_list = pols_plus_cls_list.T/prefac

    eq90_plus_pols_cls = {}
    eq90_plus_pols_cls['TT'] = pols_plus_cls_list[0]
    eq90_plus_pols_cls['EE'] = pols_plus_cls_list[1]
    # Off by minus sign
    eq90_plus_pols_cls['BB'] = -pols_plus_cls_list[2]

    # Transform back to Cl
    pols_minus_corrs = np.array([
        xi_TT/xi_mask,
        xi_p/xi_mask,   #xi_EE_camb,
        xi_eq90_minus/xi_mask,  #xi_BB_camb,
        np.zeros(lmax+1)])
    pols_minus_cls_list = camb.correlations.corr2cl(pols_minus_corrs.T,
                                            corr_x, corr_w, lmax)
    prefac[0] = 1
    pols_minus_cls_list = pols_minus_cls_list.T/prefac

    eq90_minus_pols_cls = {}
    eq90_minus_pols_cls['TT'] = pols_minus_cls_list[0]
    eq90_minus_pols_cls['EE'] = pols_minus_cls_list[1]
    # Off by minus sign
    eq90_minus_pols_cls['BB'] = pols_minus_cls_list[2]

    # Naive Polspice
    naive_pols_corrs = data_corr.T/xi_mask
    naive_pols_cls_list = camb.correlations.corr2cl(naive_pols_corrs.T,
                                                    corr_x, corr_w, lmax)
    prefac[0] = 1
    naive_pols_cls_list = naive_pols_cls_list.T/prefac

    naive_pols_cls = {}
    naive_pols_cls['TT'] = naive_pols_cls_list[0]
    naive_pols_cls['EE'] = naive_pols_cls_list[1]
    naive_pols_cls['BB'] = naive_pols_cls_list[2]

    # To vector
    plus_pols_d = np.array([])
    minus_pols_d = np.array([])
    naive_pols_d = np.array([])
    for cl_name in list(naive_pols_cls.keys()):
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
