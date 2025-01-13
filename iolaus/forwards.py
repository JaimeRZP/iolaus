# Iolaus: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023 Euclid Science Ground Segment
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
import numpy as np


def Forwards(t, M, B):
    forward_cls = {}
    for key in list(t.keys()):
        k1, k2, b1, b2 = key
        _t = t[key]
        _M = M[key]
        if k1 == k2 == "SHE":
            # Cl_EE = M_EE Cl_EE + M_BB Cl_BB
            fcls_EE = _M[0] @ _t[0] + _M[1] @ _t[1]
            # Cl_BB = M_EE Cl_BB + M_BB Cl_EE
            fcls_BB = _M[0] @ _t[1] + _M[1] @ _t[0]
            # Cl_EB = M_EB Cl_EB
            fcls_EB = _M[2] @ _t[2]
            fcls = np.array([
                B @ fcls_EE,
                B @ fcls_BB,
                B @ fcls_EB,
            ])
        else:
            fcls = (B @ (_M @ _t.T)).T
        forward_cls[key] = fcls
    return forward_cls
