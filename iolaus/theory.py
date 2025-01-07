# Iolaus: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
#
# This file is part of Iolaus.
#
# Iolaus is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iolaus is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Iolaus. If not, see <https://www.gnu.org/licenses/>.
"""The Iolaus command line interface."""
import numpy as np
import healpy as hp
import camb

def get_pars(
        H0=67.,
        omch2=0.270*0.67**2,
        ombh2=0.049*0.67**2,
        As=2.1e-9,
        ns=0.96,
        lmax=1000
        ):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, omch2=omch2, ombh2=ombh2)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.Want_CMB = False
    pars.NonLinear = camb.model.NonLinear_both
    pars.set_for_lmax(2*lmax, lens_potential_accuracy=1)
    return pars

def get_nz(z):
    nz = np.exp(-((0.3-z)/0.1)**2)
    return z, nz

def get_bz(z):
    bz = 0.83070341 + 1.19054721*z - 0.92835749*z**2 + 0.42329232*z**3
    return z, bz

def get_sources(z, nz, bz):
    sources = []
    sources += [
        camb.sources.SplinedSourceWindow(source_type='counts', z=z, W=nz, bias_z=bz),
        camb.sources.SplinedSourceWindow(source_type='lensing', z=z, W=nz)]
    return sources

def get_theory_cls(l, pars, sources):
    lmax = l.max()
    pars.SourceWindows = sources
    results = camb.get_results(pars)
    cl = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

    fl = -np.sqrt((l+2)*(l+1)*l*(l-1))
    fl /= np.clip(l*(l+1), 1, None)

    theory_cls = {}
    theory_cls[('POS', 'POS', 1, 1)] = np.array(
        cl['W1xW1'],
    )
    theory_cls[('POS', 'SHE', 1, 1)] = np.array([
        cl['W2xW1'] * fl,
        np.zeros(lmax+1),
    ])
    theory_cls[('SHE', 'SHE', 1, 1)] = np.array([
        cl['W2xW2'] * fl**2,
        np.zeros(lmax+1),
        np.zeros(lmax+1),
    ])
    return theory_cls

def theory2map(theory_cls, nside):
    tmap = hp.sphtfunc.synfast([
        theory_cls[('POS', 'POS', 1, 1)],
        theory_cls[('SHE', 'SHE', 1, 1)][0],
        theory_cls[('SHE', 'SHE', 1, 1)][1],
        theory_cls[('POS', 'SHE', 1, 1)][0]],
        nside, new=True
        )
    dict_map = {}
    dict_map[('POS', 1)] = tmap[0]
    dict_map[('SHE', 1)] = np.array([tmap[1], tmap[2]])
    return dict_map
