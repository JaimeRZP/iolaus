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


def Inversion(d, M, B, B2): 
    inversion_data_cls = {}
    for key in list(t.keys()):
        k1, k2, b1, b2 = key
        _d = d[key] @ B.T
        _M = M[key]
        if k1 == k2 == "SHE":
            _M_EB = _M[2]
            _M_EE = np.hstack((_M[0], _M[2]))
            _M_BB = np.hstack((_M[2], _M[1]))
            _M_EEBB = np.vstack((_M_EE, _M_BB))
            _inv_M_EEBB = np.linalg.pinv(_M_EEBB)
            _inv_M_EB = np.lianlg.pinv(_M_EB)
            _d_EEBB = np.hstack((_d[0], _d[1]))
            _d_EB = _d[2]
            idcls_EEBB = _d_EEBB @ _inv_M_EEBB
            idcls_EB = _d_EB @ _inv_M_EB
            idcls_EE = idcls_EEBB[:, :len(_d[0])]
            idcls_BB = idcls_EEBB[:, len(_d[0]):]
            idcls = np.array([idcls_EE, idcls_BB, idcls_EB])
        else:
            _inv_M_qq = np.linalg.pinv(_M)
            idcls = _d @ _inv_M_qq
        inversion_data_cls[key] = (B2 @ idcls).T
    return inversion_data_cls