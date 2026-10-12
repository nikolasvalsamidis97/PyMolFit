# Third-party scientific data notices

The documentation includes a public ESO HARPS Beta Pictoris observation and
a cropped SVO BT-NextGen stellar template. Their sources, model parameters
and checksums are recorded in [the example-data notice](docs/user_guide/data/README.md).
The software license does not replace the providers' data-use and citation terms.

PyMolFit includes coefficient and partition-sum tables extracted from the
LBLRTM 12.11 source distribution supplied with ESO Molecfit. The tables retain
their scientific provenance so that the corresponding pure-Python equations
can be reproduced without running or importing LBLRTM.

LBLRTM is copyright Atmospheric and Environmental Research, Inc. (AER). AER
permits downloading, installing, using, copying, and redistributing LBLRTM for
scientific and research purposes provided that its copyright notice is
reproduced and AER is appropriately acknowledged. LBLRTM or modified versions
may not be incorporated into proprietary or commercial software offered for
sale without AER's express written consent. It is supplied without express or
implied warranties. See the LBLRTM distribution and
<https://github.com/AER-RC/LBLRTM> for the authoritative license text.

Included LBLRTM-derived package data:

- `lblrtm_v12_11_tips.npz`
- `lblrtm_v12_11_h2o_continuum.npz`
- `lblrtm_v12_11_co2_continuum.npz`
- `lblrtm_v12_11_n2_fundamental.npz`
- `lblrtm_v12_11_o2_continuum.npz`
- `lblrtm_v12_11_o3_continuum.npz`
- `lblrtm_v12_11_o2_uv_continuum.npz`

The ozone and UV oxygen data are extracted by
`tools/extract_lblrtm_uv_continua.py`, which records the source SHA-256.
The ozone tables use the Chappuis/Wulf and Hartley-Huggins datasets cited
in LBLRTM `O3CH`, `BO3HH0/1/2`, and `BO3HUV` (including Molina & Molina,
1986). Far-UV oxygen uses the Lu et al. (2010) table in `bo2in_fuv`.
The Herzberg formula and pressure correction follow `HERTDA`/`HERPRS`,
including the references recorded in those routines. Regression values in
`tests/data/lblrtm_uv_reference.npz` are generated from the original Fortran
using `tools/validate_lblrtm_uv_continua.py`.

PyMolFit can also acquire the official AER line-parameter catalogue version
3.9 from Zenodo record 18881607. The 759 MiB uncompressed catalogue is not
embedded in the Python wheel: it is downloaded as versioned static data,
checksum-verified, and stored in the user's cache. An exact local copy may be
reused without duplication.

The AER catalogue copyright notice states that AER grants the right to
download, install, use, and copy the database for scientific and research
purposes. Redistribution is permitted when the copyright notice is reproduced
and AER is appropriately acknowledged. The database or a modified version may
not be incorporated into proprietary or commercial software offered for sale
without AER's express written consent, and is provided without warranties.
The full notice is installed alongside the catalogue as `AER_LICENSE.txt`.

Direct HITRAN API line data and CIA data supplied separately by users remain
subject to the HITRAN data-use terms.
