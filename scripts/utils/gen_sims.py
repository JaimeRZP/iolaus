import os
import numpy as np
import healpy as hp
import fitsio
import matplotlib.pyplot as plt
import matplotlib as mpl

import camb
from camb.sources import SplinedSourceWindow

# Resolution
nside = 1024
lmax = 1000

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

theory_cls = {}
theory_cls[('G_B', 'G_B', 0, 0)] = np.zeros(2*lmax+1)
theory_cls[('G_E', 'G_B', 0, 0)] = np.zeros(2*lmax+1)
theory_cls[('G_E', 'G_E', 0, 0)] = camb_cls['W2xW2'] * fl**2
theory_cls[('P', 'G_E', 0, 0)] = camb_cls['W2xW1'] * fl
theory_cls[('P', 'P', 0, 0)] = camb_cls['W1xW1']

# Save Cls
fname = "./sims/theory.npy"
np.save(fname, theory_cls)

for i in np.arange(1000):
    fname = "./sims/sim_{}.fits".format(i)
    if os.path.isfile(fname):
        # Already computed
        pass
    else:
        # Make map
        theory_map = hp.sphtfunc.synfast([
        theory_cls[('P', 'P', 0, 0)],
        theory_cls[('G_E', 'G_E', 0, 0)],
        theory_cls[('G_B', 'G_B', 0, 0)],
        theory_cls[('P', 'G_E', 0, 0)]],
        nside, new=True)

        # Save
        hp.fitsfunc.write_map(fname, theory_map, overwrite=True)
