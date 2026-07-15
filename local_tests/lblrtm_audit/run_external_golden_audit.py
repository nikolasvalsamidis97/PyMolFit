from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from genmolfit import (
    AtmosphereLayer,
    AtmosphereProfile,
    H2OContinuumAbsorption,
    HitranLineAbsorption,
    IsotopologueMetadata,
    LBLRTMH2OContinuum,
    LineList,
    N2ContinuumAbsorption,
    O2ContinuumAbsorption,
    PartitionTable,
    combine_optical_depth_components,
)


PROJECT = Path(__file__).resolve().parents[2]
EXTERNAL = PROJECT / "local_tests" / "external_absorption"
DEFAULT_MOLECFIT_ROOT = Path.home() / ".criresflow" / "molecfit"
DEFAULT_PROFILE = Path(
    os.environ.get(
        "GENMOLFIT_REFERENCE_ATMOSPHERE",
        PROJECT / "local_tests" / "data" / "rho01" / "ATM_PROFILE_COMBINED.fits",
    )
)
LINE_LIST = EXTERNAL / "aer_lband_h2o_co2_co_ch4_o2_strength1e-32.ecsv"
ISO_METADATA = EXTERNAL / "hitran_iso_metadata_lband.ecsv"
HITRAN_Q_DIR = EXTERNAL / "hitran_q"


@dataclass(frozen=True)
class LBLRTMCase:
    name: str
    continuum: int
    include_h2o_continuum: bool = False
    include_n2_continuum: bool = False
    include_o2_continuum: bool = False
    continuum_scales: tuple[float, ...] | None = None


CASES = (
    LBLRTMCase("h2o_lines_only", 0),
    LBLRTMCase(
        "h2o_lines_and_h2o_continuum",
        6,
        include_h2o_continuum=True,
        continuum_scales=(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    LBLRTMCase(
        "h2o_lines_and_n2_continuum",
        6,
        include_n2_continuum=True,
        continuum_scales=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ),
    LBLRTMCase(
        "h2o_lines_and_o2_continuum",
        6,
        include_o2_continuum=True,
        continuum_scales=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    ),
    LBLRTMCase("h2o_lines_and_continuum", 5, True, True, True),
)


def _run(command: Path, workdir: Path) -> None:
    completed = subprocess.run(
        [str(command)],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (workdir / f"{command.name}.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f"{command.name} failed with exit code {completed.returncode}:\n{tail}")


def _molecule_flags(*active_indices: int) -> str:
    flags = ["0"] * 47
    for index in active_indices:
        flags[index - 1] = "1"
    return "".join(flags)


def _write_lnfl_tape5(path: Path, wn_min: float, wn_max: float, flags: str) -> None:
    path.write_text(
        "$ GenMolFit external golden audit\n"
        f"{wn_min:10.3f}{wn_max:10.3f}\n"
        f"{flags:>47s}    LNOUT F100\n"
        "%%%%%%%%%%%%%%%%%%\n"
        "1234567890123456789012345678901234567890"
        "1234567890123456789012345678901234567890\n",
        encoding="ascii",
    )


def _write_lblrtm_tape5(
    path: Path,
    profile_path: Path,
    *,
    wn_min: float,
    wn_max: float,
    observer_altitude_km: float,
    zenith_angle_deg: float,
    latitude_deg: float,
    flags: str,
    continuum: int,
    continuum_scales: tuple[float, ...] | None,
    plot_spacing_cm: float,
) -> None:
    with fits.open(profile_path) as hdul:
        profile = hdul[1].data

    last_molecule = flags.rfind("1") + 1
    if last_molecule < 1:
        raise ValueError("at least one molecule must be active")
    vbar = 0.5 * (wn_min + wn_max)
    lines = [
        "$ GenMolFit external golden audit",
        (
            f"    1    1    {continuum:1d}    0    1    0    0"
            "    0    0    1    0    0    0    0    5    5    0    0"
        ),
    ]
    if continuum == 6:
        if continuum_scales is None or len(continuum_scales) != 7:
            raise ValueError("ICNTNM=6 requires seven continuum scale factors")
        lines.append(" ".join(f"{value:.8g}" for value in continuum_scales))
    lines.extend([
        f"{wn_min:10.3e}{wn_max:10.3e}{4:10d}{0.0:10.3e}{0.0:10.3e}"
        f"{2.0e-4:10.3e}{1.0e-3:10.3e}    0                  ",
        f"{0.0:10.3e}{0.0:10.3e}{0.0:10.3e}{0.0:10.3e}"
        f"{0.0:10.3e}{0.0:10.3e}{0.0:10.3e}    s",
        f"{0:5d}{3:5d}{0:5d}{0:5d}{0:5d}{last_molecule:5d}{0:5d} 0  0"
        f"{0.0:10.3e}{120.0:10.3e}{vbar:10.3e}          {latitude_deg:10.3e}",
        f"{observer_altitude_km:10.3e}{0.0:10.3e}{zenith_angle_deg:10.3e}"
        f"{0.0:10.3e}{0.0:10.3e}{0:5d}     {0.0:10.3e}",
        f"{2.0:10.3e}{5.0:10.3e}{8.0:10.3e}{0.0:10.3e}{0.0:10.3e}",
        f"{len(profile):5d}{'':8s}{'':8s}{'':8s}",
    ])

    column_names = ("H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2")
    for row in profile:
        lines.append(
            f"{float(row['HGT']):10.3e}{float(row['PRE']):10.3e}"
            f"{float(row['TEM']):10.3e}     AA   " + "A" * last_molecule
        )
        values = []
        for molecule_index in range(last_molecule):
            name = column_names[molecule_index]
            value = float(row[name]) if name in profile.names and flags[molecule_index] == "1" else 0.0
            values.append(value)
        for start in range(0, len(values), 8):
            lines.append("".join(f"{value:10.3e}" for value in values[start : start + 8]))

    lines.extend(
        [
            "-1",
            "$ Transfer to ASCII plotting data",
            " HI=0 F4=0 CN=0 AE=0 EM=0 SC=0 FI=0 PL=1 TS=0 AM=0 MG=0 LA=0 MS=0 XS=0    0    0",
            "# Plot title not used",
            f"{wn_min:10.4e}{wn_max:10.4e}{10.2:10.4e}{plot_spacing_cm:10.4e}"
            f"{1:5d}{0:5d}{12:5d}{0:5d}{1.0:10.3e}{0:2d}{0:3d}{0:5d}",
            f"{0.0:10.4g}{1.2:10.4g}{7.02:10.3e}{0.2:10.3e}"
            f"{4:5d}{0:5d}{1:5d}{0:5d}{0:5d}{0:5d}{1:2d}   {3:2d}{28:3d}",
            "-1",
            "% GenMolFit external golden audit",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _read_tape28(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    start = next(index for index, line in enumerate(lines) if "WAVENUMBER" in line) + 1
    values = []
    for line in lines[start:]:
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            values.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"no spectral rows found in {path}")
    data = np.asarray(values, dtype=float)
    return data[:, 0], data[:, 1]


def _interpolate_profile_boundary(profile, altitude_km: float, name: str) -> float:
    height = np.asarray(profile["HGT"], dtype=float)
    values = np.asarray(profile[name], dtype=float)
    return float(np.interp(altitude_km, height, values))


def _profile_to_layers(profile_path: Path, observer_altitude_km: float) -> AtmosphereProfile:
    """Convert LBLRTM boundary levels into finite, above-observer layers."""

    with fits.open(profile_path) as hdul:
        profile = hdul[1].data.copy()
    height = np.asarray(profile["HGT"], dtype=float)
    top = height[height > observer_altitude_km]
    boundaries = np.concatenate(([observer_altitude_km], top))
    species = tuple(name for name in ("H2O", "CO2", "CO", "CH4", "O2") if name in profile.names)

    boundary_values = {
        name: np.array([_interpolate_profile_boundary(profile, value, name) for value in boundaries])
        for name in ("PRE", "TEM", *species)
    }
    layers = []
    for index in range(boundaries.size - 1):
        p0 = boundary_values["PRE"][index] / 1013.25
        p1 = boundary_values["PRE"][index + 1] / 1013.25
        if p0 > 0 and p1 > 0 and not np.isclose(p0, p1):
            pressure_atm = (p0 - p1) / np.log(p0 / p1)
        else:
            pressure_atm = 0.5 * (p0 + p1)
        temperature_k = 0.5 * (
            boundary_values["TEM"][index] + boundary_values["TEM"][index + 1]
        )
        mixing = {
            name: 0.5 * (boundary_values[name][index] + boundary_values[name][index + 1]) * 1.0e-6
            for name in species
        }
        mixing["N2"] = max(0.0, 1.0 - sum(mixing.values()))
        path_length_m = (boundaries[index + 1] - boundaries[index]) * 1000.0
        layers.append(
            AtmosphereLayer(
                pressure_atm=float(pressure_atm),
                temperature_k=float(temperature_k),
                path_length_m=float(path_length_m),
                vertical_path_length_m=float(path_length_m),
                mixing_ratios=mixing,
            )
        )
    return AtmosphereProfile(tuple(layers), metadata={"level_semantics": "boundaries"})


def _genmolfit_transmission(
    wavenumber_cm: np.ndarray,
    profile_path: Path,
    observer_altitude_km: float,
    *,
    include_h2o_continuum: bool,
    include_n2_continuum: bool,
    include_o2_continuum: bool,
) -> np.ndarray:
    isotopologues = IsotopologueMetadata.from_table(ISO_METADATA)
    partitions = PartitionTable.from_hitran_q_directory(HITRAN_Q_DIR, isotopologues)
    lines = LineList.from_table(LINE_LIST).with_isotopologue_metadata(isotopologues)
    wavelength = 1.0e4 / wavenumber_cm
    lines = lines.select_range(float(np.min(wavelength)), float(np.max(wavelength)), margin=0.02)
    components = [
        HitranLineAbsorption(
            lines,
            species=("H2O",),
            partition_table=partitions,
            line_wing_mode="lblrtm_panel",
            lblrtm_sample=4.0,
            lblrtm_alfal0=0.0,
        )
    ]
    if include_h2o_continuum:
        components.append(H2OContinuumAbsorption(LBLRTMH2OContinuum.from_package_data()))
    if include_n2_continuum:
        components.append(N2ContinuumAbsorption())
    if include_o2_continuum:
        components.append(O2ContinuumAbsorption())
    atmosphere = _profile_to_layers(profile_path, observer_altitude_km)
    _, basis = combine_optical_depth_components(wavelength, atmosphere, tuple(components))
    return np.exp(-np.sum(basis, axis=0))


def _metrics(case: str, wavenumber: np.ndarray, reference: np.ndarray, model: np.ndarray) -> dict[str, float | str]:
    valid = np.isfinite(reference) & np.isfinite(model) & (reference > 0) & (model > 0)
    ref_tau = -np.log(np.clip(reference[valid], 1.0e-300, 1.0))
    model_tau = -np.log(np.clip(model[valid], 1.0e-300, 1.0))
    diff = model[valid] - reference[valid]
    reference_integral = float(np.trapz(ref_tau, wavenumber[valid]))
    reference_peak = float(np.max(ref_tau))
    return {
        "case": case,
        "n_points": int(np.count_nonzero(valid)),
        "transmission_rms": float(np.sqrt(np.mean(diff**2))),
        "transmission_max_abs": float(np.max(np.abs(diff))),
        "tau_rms": float(np.sqrt(np.mean((model_tau - ref_tau) ** 2))),
        "tau_integral_ratio": (
            float(np.trapz(model_tau, wavenumber[valid]) / reference_integral)
            if abs(reference_integral) > np.finfo(float).tiny
            else np.nan
        ),
        "tau_peak_ratio": (
            float(np.max(model_tau) / reference_peak)
            if reference_peak > np.finfo(float).tiny
            else np.nan
        ),
    }


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    molecfit_root = args.molecfit_root.resolve()
    profile_path = args.profile.resolve()
    lnfl = molecfit_root / "bin" / "lnfl"
    lblrtm = molecfit_root / "bin" / "lblrtm"
    line_database = molecfit_root / "share" / "molecfit" / "data" / "hitran" / "aer_v_3.8.1.2"
    for required in (lnfl, lblrtm, line_database, profile_path, LINE_LIST, ISO_METADATA):
        if not required.exists():
            raise FileNotFoundError(required)

    # Keep O2 active even when auditing an H2O spectral interval. LBLRTM uses
    # the active O2 amount when partitioning dry air into N2 and O2 collision
    # partners for the N2 continuum.
    flags = _molecule_flags(1, 7)
    with tempfile.TemporaryDirectory(prefix="genmolfit_lblrtm_audit_") as temporary:
        root = Path(temporary)
        lnfl_dir = root / "lnfl"
        lnfl_dir.mkdir()
        (lnfl_dir / "TAPE1").symlink_to(line_database)
        _write_lnfl_tape5(lnfl_dir / "TAPE5", args.wn_min - 5.0, args.wn_max + 5.0, flags)
        _run(lnfl, lnfl_dir)
        audit_inputs = output / "external_inputs"
        audit_inputs.mkdir(exist_ok=True)
        for name in ("TAPE5", "TAPE6", "lnfl.log"):
            source = lnfl_dir / name
            if source.exists():
                shutil.copy2(source, audit_inputs / f"lnfl_{name}")

        rows = []
        products = []
        for case in CASES:
            case_dir = root / case.name
            case_dir.mkdir()
            (case_dir / "TAPE3").symlink_to(lnfl_dir / "TAPE3")
            _write_lblrtm_tape5(
                case_dir / "TAPE5",
                profile_path,
                wn_min=args.wn_min,
                wn_max=args.wn_max,
                observer_altitude_km=args.observer_altitude_km,
                zenith_angle_deg=0.0,
                latitude_deg=args.latitude_deg,
                flags=flags,
                continuum=case.continuum,
                continuum_scales=case.continuum_scales,
                plot_spacing_cm=args.plot_spacing_cm,
            )
            _run(lblrtm, case_dir)
            for name in ("TAPE5", "TAPE6", "lblrtm.log"):
                source = case_dir / name
                if source.exists():
                    shutil.copy2(source, audit_inputs / f"{case.name}_{name}")
            wavenumber, reference = _read_tape28(case_dir / "TAPE28")
            model = _genmolfit_transmission(
                wavenumber,
                profile_path,
                args.observer_altitude_km,
                include_h2o_continuum=case.include_h2o_continuum,
                include_n2_continuum=case.include_n2_continuum,
                include_o2_continuum=case.include_o2_continuum,
            )
            rows.append(_metrics(case.name, wavenumber, reference, model))
            product = Table()
            product["wavenumber_cm-1"] = wavenumber
            product["wavelength_micron"] = 1.0e4 / wavenumber
            product["lblrtm_transmission"] = reference
            product["genmolfit_transmission"] = model
            product["difference"] = model - reference
            product.write(output / f"{case.name}.ecsv", format="ascii.ecsv", overwrite=True)
            products.append((case.name, wavenumber, reference, model))

        # Divide out the line-only calculation to isolate N2 continuum
        # transmission. This remains valid in windows where the compact
        # GenMolFit audit line list does not contain every external line.
        by_name = {name: (wn, reference, model) for name, wn, reference, model in products}
        line_wn, line_reference, line_model = by_name["h2o_lines_only"]
        n2_wn, n2_reference, n2_model = by_name["h2o_lines_and_n2_continuum"]
        if not np.array_equal(line_wn, n2_wn):
            raise RuntimeError("LBLRTM audit cases produced different wavenumber grids")
        reference_component = np.clip(n2_reference, 1.0e-300, None) / np.clip(
            line_reference, 1.0e-300, None
        )
        model_component = np.clip(n2_model, 1.0e-300, None) / np.clip(
            line_model, 1.0e-300, None
        )
        differential_name = "n2_continuum_differential"
        rows.append(
            _metrics(
                differential_name,
                line_wn,
                reference_component,
                model_component,
            )
        )
        differential = Table()
        differential["wavenumber_cm-1"] = line_wn
        differential["wavelength_micron"] = 1.0e4 / line_wn
        differential["lblrtm_transmission"] = reference_component
        differential["genmolfit_transmission"] = model_component
        differential["difference"] = model_component - reference_component
        differential.write(
            output / f"{differential_name}.ecsv", format="ascii.ecsv", overwrite=True
        )
        products.append((differential_name, line_wn, reference_component, model_component))

        o2_wn, o2_reference, o2_model = by_name["h2o_lines_and_o2_continuum"]
        if not np.array_equal(line_wn, o2_wn):
            raise RuntimeError("LBLRTM O2 audit case produced a different wavenumber grid")
        o2_reference_component = np.clip(o2_reference, 1.0e-300, None) / np.clip(
            line_reference, 1.0e-300, None
        )
        o2_model_component = np.clip(o2_model, 1.0e-300, None) / np.clip(
            line_model, 1.0e-300, None
        )
        o2_name = "o2_continuum_differential"
        rows.append(
            _metrics(o2_name, o2_wn, o2_reference_component, o2_model_component)
        )
        o2_differential = Table()
        o2_differential["wavenumber_cm-1"] = o2_wn
        o2_differential["wavelength_micron"] = 1.0e4 / o2_wn
        o2_differential["lblrtm_transmission"] = o2_reference_component
        o2_differential["genmolfit_transmission"] = o2_model_component
        o2_differential["difference"] = o2_model_component - o2_reference_component
        o2_differential.write(output / f"{o2_name}.ecsv", format="ascii.ecsv", overwrite=True)
        products.append((o2_name, o2_wn, o2_reference_component, o2_model_component))

    summary = Table(rows=rows)
    summary.write(output / "summary.ecsv", format="ascii.ecsv", overwrite=True)
    summary.write(output / "summary.csv", format="ascii.csv", overwrite=True)

    fig, axes = plt.subplots(len(products), 2, figsize=(12, 4 * len(products)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, (name, wavenumber, reference, model) in enumerate(products):
        axes[row, 0].plot(wavenumber, reference, color="tab:blue", lw=0.9, label="LBLRTM 12.11")
        axes[row, 0].plot(wavenumber, model, color="tab:orange", lw=0.8, label="GenMolFit")
        axes[row, 0].set_ylabel("Transmission")
        axes[row, 0].set_title(name)
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].plot(wavenumber, model - reference, color="black", lw=0.8)
        axes[row, 1].axhline(0.0, color="0.6", lw=0.7)
        axes[row, 1].set_ylabel("GenMolFit - LBLRTM")
        axes[row, 1].set_title(f"{name}: residual")
        axes[row, 0].set_xlabel("Wavenumber [cm$^{-1}$]")
        axes[row, 1].set_xlabel("Wavenumber [cm$^{-1}$]")
    fig.savefig(output / "external_lblrtm_golden_audit.png", dpi=180)
    plt.close(fig)
    print(summary)
    print(f"Wrote {output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--molecfit-root", type=Path, default=DEFAULT_MOLECFIT_ROOT)
    result.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    result.add_argument("--output", type=Path, default=PROJECT / "local_tests" / "lblrtm_external_golden")
    result.add_argument("--wn-min", type=float, default=2496.0)
    result.add_argument("--wn-max", type=float, default=2506.0)
    result.add_argument("--plot-spacing-cm", type=float, default=0.005)
    result.add_argument("--observer-altitude-km", type=float, default=2.635)
    result.add_argument("--latitude-deg", type=float, default=-24.6276)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
