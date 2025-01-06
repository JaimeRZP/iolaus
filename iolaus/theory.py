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
import camb

def get_pars():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67., omch2=0.270*0.67**2, ombh2=0.049*0.67**2)
    pars.InitPower.set_params(As=2.1e-9, ns=0.96)
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
    pars.SourceWindows = sources
    results = camb.get_results(pars)
    cl = results.get_source_cls_dict(sources, lmax=lmax)

    fl = -np.sqrt((l+2)*(l+1)*l*(l-1))
    fl /= np.clip(l*(l+1), 1, None)

    theory_cls = {}
    theory_cls[('SHE', 'SHE')] = np.array([
        cl['W2xW2'] * fl**2,
        np.zeros(lmax+1),
        np.zeros(lmax+1),
    ])
    theory_cls[('POS', 'SHE')] = np.array([
        cl['W2xW1'] * fl,
        np.zeros(lmax+1),
    ])
    theory_cls[('POS', 'POS')] = np.array([
        cl['W1xW1'],
    ])
    return cl
