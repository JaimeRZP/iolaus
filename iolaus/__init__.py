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
"""
Main module of the *Iolaus* package.
"""

__all__ = [
    # utils
    "make_fields",
    "get_cls",
    "bin2pt"
    "make_mask",
    "mask2cls",
    "data2cls",
    "apply_mask",
    "compsep_cls"
    # polspice_utils
    "legendre_funcs",
    "cl2corr",
    "corr2cl",
    # theory
    "get_pars",
    "get_nz",
    "get_bz",
    "get_sources",
    "get_theory_cls",
    "theory2map",
    # forwards
    "Forwards",
    # inversion
    "Inversion",
    # master
    "Master",
    # polspice
    "Polspice",
    "Naive_Polspice",
    "Eq90",
]

from .utils import (
    make_fields,
    get_cls,
    bin2pt,
    make_mask,
    mask2cls,
    data2cls,
    apply_mask,
    compsep_cls
)

from .theory import (
    get_pars,
    get_nz,
    get_bz,
    get_sources,
    get_theory_cls,
    theory2map,
)

from .polspice_utils import (
    legendre_funcs,
    cl2corr,
    corr2cl,
)

from .forwards import (
    Forwards,
)

from .inversion import (
    Inversion,
)

from .master import (
    Master,
)

from .polspice import (
    Naive_Polspice,
    Polspice,
    Eq90,
)
