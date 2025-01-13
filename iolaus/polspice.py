# DICES: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
#
# This file is part of DICES.
#
# DICES is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DICES is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with DICES. If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from .polspice_utils import cl2corr, corr2cl, l2x
from scipy.interpolate import interp1d

def Naive_Polspice(d, m, B, patch_hole=True):
    corr_d = {}
    for key in list(d.keys()):
        k1, k2, b1, b2 = key
        _d = d[key]
        # Correct Cl by mask
        if k1 == k2 == "SHE":
            _m = m[('WHT', 'WHT', b1, b2)]
            __m = np.array([
                    _m,
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                ])
            wm = cl2corr(__m.T).T[0]
            if patch_hole:
                print("Patching holes")
                wm *= _logistic(np.log10(np.abs(wm)))
            if b1 == b2:
                __d = np.array([
                        np.zeros_like(_d[0]),
                        _d[0],  # EE like spin-2
                        _d[1],  # BB like spin-2
                        np.zeros_like(_d[0]),
                    ])
                __id = np.array(
                    [
                        np.zeros_like(_d[0]),
                       -_d[2],  # EB like spin-0
                        _d[2],   # EB like spin-0
                        np.zeros_like(_d[0]),
                    ]
                )
                # Transform to real space
                w = cl2corr(__d.T).T + 1j * cl2corr(__id.T).T
                # Correct by mask
                corr_w = (w / wm).real
                icorr_w = (w / wm).imag
                # Transform back to Cl
                __corr_d = corr2cl(corr_w.T).T
                __icorr_d = corr2cl(icorr_w.T).T
                _corr_d = np.array(
                    [
                        B @ __corr_d[1],  # EE like spin-2
                       -B @ __corr_d[2],  # BB like spin-2
                        B @ __icorr_d[1],  # EB like spin-0
                    ]
                )
            if b1 != b2:
                __d = np.array(
                    [
                        np.zeros_like(_d[0]),
                        _d[0],  # EE like spin-2
                        _d[1],  # BB like spin-2
                        np.zeros_like(_d[0]),
                    ]
                )
                __id = np.array(
                    [
                        np.zeros_like(_d[0]),
                        -_d[2],  # EB like spin-0
                        _d[3],   # BE like spin-0
                        np.zeros_like(_d[0]),
                    ]
                )
                # Correct by alpha
                wd = cl2corr(__d.T).T + 1j * cl2corr(__d.T).T
                corr_wd = (wd / wm).real
                icorr_wd = (wd / wm).imag
                # Transform back to Cl
                __corr_d = corr2cl(corr_wd.T).T
                __icorr_d = corr2cl(icorr_wd.T).T
                _corr_d = np.array(
                    [
                        B @ __corr_d[1],    # EE like spin-2
                       -B @ __corr_d[2],    # BB like spin-2
                        B @ __icorr_d[1],   # EB like spin-0
                       -B @ __icorr_d[2],  # BE like spin-0
                    ]
                )
        if k1 == k2 == "POS":
            # Treat everything as spin-0
            _m = m[('VIS', 'VIS', b1, b2)]
            __m = np.array([
                    _m,
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                ])
            wm = cl2corr(__m.T).T[0]
            if patch_hole:
                print("Patching holes")
                wm *= _logistic(np.log10(np.abs(wm)))
            __d = np.array([
                    _d,
                    np.zeros_like(_d),
                    np.zeros_like(_d),
                    np.zeros_like(_d),
                ])
            # Correct by mask
            wd = cl2corr(__d.T).T
            corr_wd = wd / wm
            # Transform back to Cl
            __corr_d = corr2cl(corr_wd.T).T
            _corr_d = B @ __corr_d[0]
        if k1 != k2:
            _m = m[('VIS', 'WHT', b1, b2)]
            __m = np.array([
                    _m,
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                ])
            wm = cl2corr(__m.T).T[0]
            if patch_hole:
                print("Patching holes")
                wm *= _logistic(np.log10(np.abs(wm)))
            _corr_d = []
            for cl in _d:
                __d = np.array([
                        cl,
                        np.zeros_like(cl),
                        np.zeros_like(cl),
                        np.zeros_like(cl),
                    ])
                # Correct by mask
                wd = cl2corr(__d.T).T
                corr_wd = wd / wm
                # Transform back to Cl
                __corr_d = corr2cl(corr_wd.T).T
                _corr_d.append(B @ __corr_d[0])
            _corr_d = np.array(_corr_d)
        corr_d[key] = _corr_d
    return corr_d

def Polspice(d, m, B, patch_hole=True):
    corr_d = {}
    for key in list(d.keys()):
        k1, k2, b1, b2 = key
        _d = np.atleast_2d(d[key])
        # Correct Cl by mask
        if k1 == k2 == "SHE":
            _m = m[('WHT', 'WHT', b1, b2)]
            __m = np.array(
                [
                    _m,
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                ]
            )
            wm = cl2corr(__m.T).T
            if patch_hole:
                wm *= _logistic(np.log10(np.abs(wm)))
            if b1 == b2:
                __d = np.array(
                    [
                        np.zeros_like(_d[0]),
                        _d[0],  # EE like spin-2
                        _d[1],  # BB like spin-2
                        np.zeros_like(_d[0]),
                    ]
                )
                __id = np.array(
                    [
                        np.zeros_like(_d[0]),
                        -_d[2],  # EB like spin-0
                        _d[2],  # EB like spin-0
                        np.zeros_like(_d[0]),
                    ]
                )
                # Transform to real space
                w = cl2corr(__d.T).T + 1j * cl2corr(__id.T).T
                # Correct by mask
                lmax = len(_d[0]) - 1
                xvals, weights = l2x(lmax)
                corr_w = (w / wm).real
                icorr_w = (w / wm).imag
                eq90 = Eq90(xvals, corr_w[1])
                corr_w = np.array([
                    corr_w[0],
                    corr_w[1],
                    eq90,
                    corr_w[3],
                    ])
                # Transform back to Cl
                __corr_d = corr2cl(corr_w.T).T
                __icorr_d = corr2cl(icorr_w.T).T
                _corr_d = np.array(
                    [
                        B @ __corr_d[1],  # EE like spin-2
                       -B @ __corr_d[2],  # BB like spin-2
                        B @ __icorr_d[1],  # EB like spin-0
                    ]
                )
            if b1 != b2:
                __d = np.array(
                    [
                        np.zeros_like(_d[0]),
                        _d[0],  # EE like spin-2
                        _d[1],  # BB like spin-2
                        np.zeros_like(_d[0]),
                    ]
                )
                __id = np.array(
                    [
                        np.zeros_like(_d[0]),
                       -_d[2],  # EB like spin-0
                        _d[3],   # BE like spin-0
                        np.zeros_like(_d[0]),
                    ]
                )
                # Correct by mask
                wd = cl2corr(__d.T).T + 1j * cl2corr(__d.T).T
                corr_wd = (wd / wm).real
                icorr_wd = (wd / wm).imag
                # Transform back to Cl
                __corr_d = corr2cl(corr_wd.T).T
                __icorr_d = corr2cl(icorr_wd.T).T
                _corr_d = np.array(
                    [
                        B @ __corr_d[1],    # EE like spin-2
                       -B @ __corr_d[2],    # BB like spin-2
                        B @ __icorr_d[1],   # EB like spin-0
                       -B @ __icorr_d[2],  # BE like spin-0
                    ]
                )
        if k1 == k2 == "POS":
            # Treat everything as spin-0
            _m = m[('VIS', 'VIS', b1, b2)]
            __m = np.array(
                [
                    _m,
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                ]
            )
            wm = cl2corr(__m.T).T
            if patch_hole:
                wm *= _logistic(np.log10(np.abs(wm)))
            __d = np.array(
                [
                    _d,
                    np.zeros_like(_d),
                    np.zeros_like(_d),
                    np.zeros_like(_d),
                ]
            )
            # Correct by mask
            wd = cl2corr(__d.T).T
            corr_wd = wd / wm
            # Transform back to Cl
            __corr_d = corr2cl(corr_wd.T).T
            _corr_d = np.array(B @ __corr_d[0])
        if k1 != k2:
            _m = m[('VIS', 'WHT', b1, b2)]
            __m = np.array(
                [
                    _m,
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                    np.zeros_like(_m),
                ]
            )
            wm = cl2corr(__m.T).T
            if patch_hole:
                wm *= _logistic(np.log10(np.abs(wm)))
            _corr_d = []
            for cl in _d:
                __d = np.array(
                    [
                        cl,
                        np.zeros_like(cl),
                        np.zeros_like(cl),
                        np.zeros_like(cl),
                    ]
                )
                wd = cl2corr(__d.T).T
                # Correct by mask
                corr_wd = wd / wm
                # Transform back to Cl
                __corr_d = corr2cl(corr_wd.T).T
                _corr_d.append(B @ __corr_d[0])
            _corr_d = np.array(_corr_d)
        corr_d[key] = _corr_d
    return corr_d

def Eq90(cos_theta, xi_p):
    xi_pi = interp1d(cos_theta, xi_p, kind='linear', fill_value="extrapolate")
    #x = np.linspace(-0.9999, 0.9995, 101)
    x = np.linspace(cos_theta[0], cos_theta[-1], 100000000)
    xi_p = xi_pi(x)
    dx = x[1] - x[0]
    eps = 1e-5
    prefac = 8/((1-x + eps)**2)
    prefac1 = (1-x)

    integ1 = dx * xi_p * (1/((1+x+eps)**2))
    int1 = np.cumsum(integ1[::-1])[::-1]
    int1 = np.append(int1[1:], 0)
    t1 = prefac1 * int1

    prefac2 = (2+x)
    integ2 = dx * xi_p * ((1-x**2)/((1+x+eps)**3))
    int2 = np.cumsum(integ2[::-1])[::-1]
    int2 = np.append(int2[1:], 0)
    t2 = prefac2 * int2

    eq90 = xi_p + prefac *(t1 - t2)
    eq90_i = interp1d(x, eq90, kind='linear', fill_value="extrapolate")
    eq90 = eq90_i(cos_theta)
    return eq90

def _logistic(x, x0=-2, k=50):
        return 1.0 + np.exp(-k * (x - x0))
