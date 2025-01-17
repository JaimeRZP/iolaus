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
from heracles.fields import Positions, Shears, Visibility, Weights
from heracles import transform, angular_power_spectra
from heracles.healpy import HealpixMapper
from copy import deepcopy

def make_fields(mapper, mode='data'):
    She_lonlat = ('SHE_RA', 'SHE_DEC')
    Pos_lonlat = ('SHE_RA', 'SHE_DEC')
    if mode=='data':
        fields = {
            "POS": Positions(
                mapper,
                *Pos_lonlat,
                mask="VIS"
                ),
            "SHE": Shears(
                mapper,
                *She_lonlat,
                'SHE_E1_CAL',
                'SHE_E2_CAL',
                'SHE_WEIGHT',
                mask="WHT",
            ),
        }
    if mode=='vis':
        fields = {
            "VIS": Visibility(mapper),
            "WHT": Weights(
                mapper,
                *She_lonlat,
                'SHE_WEIGHT'
                ),
        }
    return fields

def get_cls(data_maps, vis_maps, mask,
             nside=1024, lmax=1000):
    """
    Internal method to compute the Cls.
    """
    # Deep copy to avoid modifying the original maps
    _mask = np.copy(mask)
    data = deepcopy(data_maps)
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields_data = make_fields(mapper, mode='data')
    fields_vis  = make_fields(mapper, mode='vis')
    for key in data.keys():
        if key[0] == "POS":
            data[key] *= _mask
        else:
            data[key][0] *= _mask
            data[key][1] *= _mask
    # compute to alms
    alms = transform(fields_data, data)
    # compute cls
    cls = angular_power_spectra(alms)

    data_mm = deepcopy(vis_maps)
    for key in data_mm.keys():
        data_mm[key] *= _mask
    alms_mm = transform(fields_vis, data_mm)
    cls_mm = angular_power_spectra(alms_mm)
    return cls, cls_mm

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

    return binned

def make_mask(nside, mode='patch'):
    mask = np.ones(hp.nside2npix(nside))
    pixel_theta, pixel_phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    if mode == 'patch':
        mask[np.pi/3 > pixel_theta] = 0.0
        mask[pixel_theta > 2*np.pi/3] = 0.0
        mask[pixel_phi > np.pi/2] = 0.0
        mask[np.pi/8> pixel_phi] = 0.0
    if mode == "1/3":
        mask[np.pi/3 > pixel_theta] = 0.0
    if mode == "1/2":
        mask[pixel_theta > np.pi/2] = 0.0
    if mode == "2/3":
        mask[2*np.pi/3 > pixel_theta] = 0.0
    else:
        pass
    mask_dict = {}
    mask_dict[("VIS", 1)] = mask
    mask_dict[("WHT", 1)] = mask
    return mask_dict

def mask2cls(mask, nside, lmax):
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields = make_fields(mapper, mode='vis')
    alms = transform(fields, mask)
    cls = angular_power_spectra(alms)
    return cls

def data2cls(data_map, nside, lmax):
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields = make_fields(mapper, mode='data')
    alms = transform(fields, data_map)
    cls = angular_power_spectra(alms)
    return cls

def apply_mask(data_map, mask):
    masked_map = {}
    for i in range(len(list(data_map.keys()))):
        key_map = list(data_map.keys())[i]
        key_mask = list(mask.keys())[i]
        m = data_map[key_map]
        mm = mask[key_mask]
        masked_map[key_map] = m * mm
    return masked_map

def compsep_cls(Cls):
    """
    Separates the SHE values into E and B modes.
    input:
        Cls: dictionary of Cl values
    returns:
        Cls_unraveled: dictionary of Cl values
    """
    Cls_compsep = {}
    for key in list(Cls.keys()):
        t1, t2, b1, b2 = key
        cl = np.atleast_2d(Cls[key])
        if t1 == t2 == "POS":
            Cls_compsep[key] = cl[..., 0, :]
        elif t1 == t2 == "SHE" and b1 == b2:
            Cls_compsep[("G_E", "G_E", b1, b2)] = cl[..., 0, :]
            Cls_compsep[("G_B", "G_B", b1, b2)] = cl[..., 1, :]
            Cls_compsep[("G_E", "G_B", b1, b2)] = cl[..., 2, :]
        elif t1 == t2 == "SHE" and b1 != b2:
            Cls_compsep[("G_E", "G_E", b1, b2)] = cl[..., 0, :]
            Cls_compsep[("G_B", "G_B", b1, b2)] = cl[..., 1, :]
            Cls_compsep[("G_E", "G_B", b1, b2)] = cl[..., 2, :]
            Cls_compsep[("G_E", "G_B", b2, b1)] = cl[..., 3, :]
        elif t1 == "POS" and t2 == "SHE":
            Cls_compsep[("POS", "G_E", b1, b2)] = cl[..., 0, :]
            Cls_compsep[("POS", "G_B", b1, b2)] = cl[..., 1, :]
    return Cls_compsep
