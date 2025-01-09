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
"""The Iolaus command line interface."""
import numpy as np
import .forwards as fwd


def Master(d, t, M, B, B2): 
    ft = fwd.forwards(t, M, B2)
    master_data_cls = {}
    master_theory_cls = {}
    for key in list(t.keys()):
        k1, k2, b1, b2 = key
        _d = d[key] @ B.T
        _ft = ft[key]
        _M = M[key]
        _M_qq = B @ _M @ B2.T
        if k1 == k2 == "SHE":
            _M_qq_EB = _M_qq[2]
            _M_qq_EE = np.hstack((_M_qq[0], _M_qq[2]))
            _M_qq_BB = np.hstack((_M_qq[2], _M_qq[1]))
            _M_qq_EEBB = np.vstack((_M_qq_EE, _M_qq_BB))
            _inv_M_qq_EEBB = np.linalg.pinv(_M_qq_EEBB)
            _inv_M_qq_EB = np.lianlg.pinv(_M_qq_EB)
            _d_EEBB = np.hstack((_d[0], _d[1]))
            _t_EEBB = np.hstack((_ft[0], _ft[1]))
            _d_EB = _d[2]
            _t_EB = _ft[2]
            mdcls_EEBB = _d_EEBB @ _inv_M_qq_EEBB
            mdcls_EB = _d_EB @ _inv_M_qq_EB
            mtcls_EEBB = _t_EEBB @ _inv_M_qq_EEBB
            mtcls_EB = _t_EB @ _inv_M_qq_EB
            mdcls_EE = mdcls_EEBB[:, :len(_d[0])]
            mdcls_BB = mdcls_EEBB[:, len(_d[0]):]
            mtcls_EE = mtcls_EEBB[:, :len(_ft[0])]
            mtcls_BB = mtcls_EEBB[:, len(_ft[0]):]
            mdcls = np.array([mdcls_EE, mdcls_BB, mdcls_EB])
            mtcls = np.array([mtcls_EE, mtcls_BB, mtcls_EB])
        else:
            _inv_M_qq = np.linalg.pinv(_M_qq)
            mdcls = _d @ _inv_M_qq
            mtcls = _ft @ _inv_M_qq
        master_data_cls[key] = mdcls
        master_theory_cls[key] = mtcls
    return master_data_cls, master_theory_cls
