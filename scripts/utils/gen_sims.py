import os
import heracles
from heracles.healpy import HealpixMapper
from  heracles import Positions, Shears, transform, angular_power_spectra
import numpy as np
import healpy as hp
import fitsio
import matplotlib.pyplot as plt
import matplotlib as mpl

import camb
from camb.sources import SplinedSourceWindow

# Resolution
nside = 1024
lmax = 1500

# Parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=67., omch2=0.270*0.67**2, ombh2=0.049*0.67**2)
pars.InitPower.set_params(As=2.1e-9, ns=0.96)
pars.Want_CMB = False 
pars.NonLinear = camb.model.NonLinear_both
pars.set_for_lmax(2*lmax, lens_potential_accuracy=1);

# Tomography
z = np.linspace(0, 3, 300)
nz = np.exp(-((0.3-z)/0.1)**2)
bz = 0.83070341 + 1.19054721*z - 0.92835749*z**2 + 0.42329232*z**3
sources = []
sources += [
    SplinedSourceWindow(source_type='counts', z=z, W=nz, bias_z=bz),
    SplinedSourceWindow(source_type='lensing', z=z, W=nz)]
pars.SourceWindows = sources

# Cls
results = camb.get_results(pars)
camb_cls = results.get_source_cls_dict(lmax=2*lmax, raw_cl=True)

l = np.arange(2*lmax+1)
fl = -np.sqrt((l+2)*(l+1)*l*(l-1))
fl /= np.clip(l*(l+1), 1, None)

theory = {}
# get the full-sky spectra; B-mode is assumed zero
cl_pp = camb_cls[f"W{2 * i - 1}xW{2 * j - 1}"]
cl_pe = fl * camb_cls[f"W{2 * i - 1}xW{2 * j}"]
cl_pb = np.zeros_like(cl_pe)
cl_ep = fl * camb_cls[f"W{2 * i}xW{2 * j - 1}"]
cl_bp = np.zeros_like(cl_ep)
cl_ee = fl**2 * camb_cls[f"W{2 * i}xW{2 * j}"]
cl_bb = np.zeros_like(cl_ee)
cl_eb = np.zeros_like(cl_ee)
cl_be = np.zeros_like(cl_ee)

# all mixing matrix combinations
key = ("POS", "POS", 0, 0)
cl = np.array(cl_pp)
theory[key] = heracles.Result(cl, axis=(0,))

key = ("POS", "SHE", 0, 0)
cl = np.array([cl_ep, cl_bp])
theory[key] = heracles.Result(cl, axis=(1,))

key = ("SHE", "SHE", 0, 0)
cl = np.array(
    [
        [cl_ee, cl_eb],
        [cl_be, cl_bb],
    ]
)
theory[key] = heracles.Result(cl, axis=(2,))

# Save Cls
heracles.write("./sims/theory_cls.fits", theory, clobber=True)

for i in np.arange(1000):
    fname = "./sims/sim_{}.fits".format(i)
    if os.path.isfile(fname):
        # Already computed
        pass
    else:
        # Make map
        theory_map = hp.sphtfunc.synfast([
        theory[('POS', 'POS', 0, 0)],
        theory[('SHE', 'SHE', 0, 0)][0, 0],
        theory[('SHE', 'SHE', 0, 0)][1, 1],
        theory[('POS', 'SHE', 0, 0)]],
        nside, new=True)

        mapper = HealpixMapper(nside, lmax)
        fields = {
            "P": Positions(mapper, "RIGHT_ASCENSION", "DECLINATION", mask="V"),
            "G": Shears(mapper, "RIGHT_ASCENSION", "DECLINATION", "G1", "-G2",
                        "WEIGHT", mask="W")}

        maps = {('P', 0): theory_map[0]*mask,
                ('G', 0): np.array([theory_map[1]*mask, theory_map[2]*mask], np.dtype(float, metadata={'spin': 2})),}
        alms = transform(fields, maps, progress=False)

        # Two-point Statistics
        data_cls = angular_power_spectra(alms)

        # Save
        data_cls = {}

