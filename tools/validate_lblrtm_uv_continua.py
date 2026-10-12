"""Compare Python ozone/O2-UV opacity against the original LBLRTM routines.

Usage: python tools/validate_lblrtm_uv_continua.py /path/to/lblrtm/src
Requires gfortran only for this developer validation, never at runtime.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pymolfit import LBLRTMO3Continuum
from pymolfit.continuum import _o2_uv_cross_section

DRIVER = """
module lblparams
  integer, parameter :: n_absrb=100000
end module
program reference
  use lblparams
  implicit real*8(v)
  common /ABSORB/ v1abs,v2abs,dvabs,nptabs,absrb(n_absrb)
  common /LAMCHN/ onepl,onemi,expmin,argmin
  dimension c(10000),c0(10000),c1(10000),c2(10000)
  onepl=1.001
  onemi=0.999
  read(*,*) branch,v1abs,v2abs,dvabs,t,p,jrad
  nptabs=nint((v2abs-v1abs)/dvabs)+1
  absrb=0.
  dt=t-273.15
  ib=nint(branch)
  select case(ib)
  case(1)
    call XO3CHP(v1c,v2c,dvc,n,c0,c1,c2,v1ss,v2ss)
    c(:n)=(c0(:n)+dt*(c1(:n)+dt*c2(:n)))*1.e-2
  case(2)
    call O3HHT0(v1c,v2c,dvc,n,c0,v1ss,v2ss)
    call O3HHT1(v1t,v2t,dvt,nt,c1)
    call O3HHT2(v1t,v2t,dvt,nt,c2)
    c(:n)=c0(:n)*(1.+dt*(c1(:n)+dt*c2(:n)))*1.e-2
  case(3)
    call O3HHUV(v1c,v2c,dvc,n,c0,v1ss,v2ss)
    c(:n)=c0(:n)*1.e18
  case(4,5)
    call O2HERZ(v1c,v2c,dvc,n,c0,t,p,v1ss,v2ss)
    c(:n)=c0(:n)*1.e3
  end select
  do j=1,n
    vj=v1c+dvc*real(j-1)
    if(jrad.eq.1) c(j)=c(j)*RADFN(vj,t/1.4387752)
  enddo
  call XINT(v1c,v2c,dvc,c,1.,v1abs,dvabs,absrb,1,nptabs)
  if(ib.eq.5) then
    call O2FUV(v1c,v2c,dvc,n,c0,v1ss,v2ss)
    c(:n)=c0(:n)*1.e3
    do j=1,n
      vj=v1c+dvc*real(j-1)
      if(jrad.eq.1) c(j)=c(j)*RADFN(vj,t/1.4387752)
    enddo
    call XINT(v1c,v2c,dvc,c,1.,v1abs,dvabs,absrb,1,nptabs)
  endif
  do j=1,nptabs
    write(*,'(ES24.15)') absrb(j)
  enddo
end program
"""


def _routine(text: str, kind: str, name: str) -> str:
    match = re.search(
        rf"^{kind}\s+{name}\b.*?^end\s+{kind}\s+{name}\b[^\n]*",
        text,
        re.I | re.S | re.M,
    )
    if match is None:
        raise ValueError(f"missing {kind} {name}")
    return match[0]


def validate(source: Path, reference_output: Path | None = None) -> None:
    contnm = (source / "contnm.f90").read_text()
    oprop = (source / "oprop.f90").read_text()
    routines = [DRIVER]
    routines.extend(
        _routine(contnm, "subroutine", name)
        for name in (
            "XO3CHP",
            "O3HHT0",
            "O3HHT1",
            "O3HHT2",
            "O3HHUV",
            "O2HERZ",
            "HERTDA",
            "HERPRS",
            "O2FUV",
        )
    )
    routines.extend(
        _routine(contnm, "block data", name)
        for name in (
            "O3CH",
            "BO3HH0",
            "BO3HH1",
            "BO3HH2",
            "BO3HUV",
            "bo2in_fuv",
        )
    )
    routines.extend((_routine(oprop, "subroutine", "XINT"), _routine(oprop, "function", "RADFN")))
    ozone = LBLRTMO3Continuum.from_package_data()
    worst = 0.0
    cases = 0
    negative_roundoff = 0.0
    records = []
    with tempfile.TemporaryDirectory(prefix="pymolfit-uv-reference-") as directory:
        directory = Path(directory)
        (directory / "reference.f90").write_text("\n".join(routines))
        subprocess.run(
            [
                "gfortran",
                "-w",
                "-fdefault-real-8",
                "-fdefault-double-8",
                "-ffree-line-length-none",
                "reference.f90",
                "-o",
                "reference",
            ],
            cwd=directory,
            check=True,
            capture_output=True,
        )
        # Non-grid-aligned samples exercise the original XINT as well as tables.
        bands = (
            (1, 8920.25, 24664.25),
            (2, 27370.25, 40799.25),
            (3, 40800.25, 53999.25),
            (4, 36000.25, 56739.25),
            (5, 56740.25, 86959.25),
        )
        for branch, first, last in bands:
            grid = np.arange(first, last + 0.5, 1.0)
            for temperature in (200.0, 250.0, 296.0):
                for pressure in (100.0, 1013.0):
                    for jrad in (0, 1):
                        result = subprocess.run(
                            [str(directory / "reference")],
                            check=True,
                            capture_output=True,
                            text=True,
                            input=f"{branch} {first} {last} 1 {temperature} {pressure} {jrad}\n",
                        )
                        reference = np.loadtxt(io.StringIO(result.stdout))
                        if branch <= 3:
                            actual = ozone.optical_depth_layer(
                                grid,
                                o3_column_cm2=1e18,
                                temperature_k=temperature,
                                jrad=jrad,
                            )
                        else:
                            actual = 1e23 * _o2_uv_cross_section(
                                grid,
                                temperature_k=temperature,
                                pressure_mbar=pressure,
                                jrad=jrad,
                            )
                        # XINT can undershoot zero at sharp troughs and band
                        # edges. PyMolFit forbids negative absorption there.
                        negative_roundoff = min(negative_roundoff, float(np.min(reference)))
                        reference = np.maximum(reference, 0.0)
                        np.testing.assert_allclose(actual, reference, rtol=2e-9, atol=1e-15)
                        significant = np.abs(reference) > max(
                            np.max(np.abs(reference)) * 1e-10, 1e-30
                        )
                        worst = max(
                            worst,
                            float(np.max(np.abs(actual[significant] / reference[significant] - 1))),
                        )
                        records.extend(
                            [branch, temperature, pressure, jrad, grid[index], reference[index]]
                            for index in (0, grid.size // 4, grid.size // 2, grid.size - 1)
                        )
                        cases += 1
        print(f"Passed {cases} Fortran comparisons; worst relative difference {worst:.3g}")
        print(
            f"Most negative reference interpolation value clipped to zero: {negative_roundoff:.3g}"
        )
    if reference_output is not None:
        reference_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reference_output,
            cases=np.asarray(records),
            source=np.asarray("LBLRTM 12.11 routines + XINT/RADFN; gfortran real-8"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()
    validate(args.source, args.reference_output)
