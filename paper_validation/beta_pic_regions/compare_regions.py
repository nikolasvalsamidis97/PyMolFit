"""Na D before/after correction for Beta Pic HARPS and AU Mic ESPRESSO.

Run directly to save and open regions_mosaic.png. Beta Pic uses the saved
full-spectrum fit. AU Mic is fitted in 592.05--592.22 nm and evaluated across
586--594 nm. Fits are cached; --plot-only forbids fitting, --refit-aumic repeats it.
--no-show saves the single PNG without opening a window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Do not create bytecode folders beside the paper scripts.
sys.dont_write_bytecode = True
from paper_validation.beta_pic_full import compare_full as full  # noqa: E402

DEFAULT_SOURCE = full.DEFAULT_DATA
DEFAULT_AUMIC_SOURCE = REPO / "validation_results/aumic_nad_continua"
DEFAULT_AUMIC_INPUT = (
    REPO.parent / "testscripts/validation_spectra/AUMic/Observed/HD197481_ESPRESSO_SN79.fits"
)
DEFAULT_AUMIC_TEMPLATE = (
    Path.home()
    / "LocalProjects/Alexis_AUMic_project/RadiationPressure/Spectra/au_mic/Model"
    / "au_mic_btsettl_cifist_svo_fid205_teff3600_logg4p0_feh0p0.txt"
)
AUMIC_PROTOCOL = "aumic_espresso_nad_continua_v2"
# Display windows in observatory-frame vacuum nm, not additional fitting regions.
REGIONS = {
    "nad": ("Na I D doublet", (588.0, 591.0)),
}
COLORS = {"pymolfit": "#0072B2", "molecfit": "#D55E00"}


def load_products(source, protocol=full.COMPARISON_PROTOCOL):
    """Require matching all-species corrections; no stellar reference is plotted."""
    provenance = json.loads((source / "provenance.json").read_text())
    if provenance.get("comparison_protocol") != protocol:
        raise ValueError("Use the full-spectrum results_all_species products, not the old Na D fit")
    if provenance["pymolfit_settings"].get("hitran_species") is not None:
        raise ValueError(
            "The source correction restricts species; an unrestricted AER run is required"
        )
    py = Table.read(source / "pymolfit.ecsv")
    if not py.meta["fit_success"]:
        raise ValueError("The PyMolFit source fit did not converge")
    policy = full.molecular_policy(
        SimpleNamespace(
            species_scales=json.loads(py.meta["species_scales"]),
            parameter_names=json.loads(py.meta["parameter_names"]),
            provenance=json.loads(py.meta["provenance_json"]),
        )
    )
    for key in ("included", "fitted", "fixed"):
        if policy[key] != provenance["species_policy"][key]:
            raise ValueError("Saved molecular policy differs from the actual PyMolFit product")
    molecules = Table.read(source / "molecfit/MODEL_MOLECULES.fits", hdu=1)
    included = [str(name).strip() for name in molecules["LIST_MOLEC"]]
    fitted = [name for name, flag in zip(included, molecules["FIT_MOLEC"], strict=True) if flag]
    if sorted(included) != policy["included"] or sorted(fitted) != policy["fitted"]:
        raise ValueError("Molecfit and PyMolFit did not include/fit the same species")
    for name, scale in zip(included, molecules["REL_COL"], strict=True):
        if name in policy["fixed"] and not np.isclose(scale, policy["fixed"][name]):
            raise ValueError("Molecfit changed a fixed molecular abundance")

    table = Table.read(source / "comparison.ecsv")
    return table, provenance


def ensure_aumic(source, input_path, template_path, *, refit=False, plot_only=False):
    if refit and plot_only:
        raise ValueError("--refit-aumic and --plot-only cannot be combined")
    if not refit and (source / "comparison.ecsv").exists():
        table, provenance = load_products(source, AUMIC_PROTOCOL)
        if full.sha256(input_path) != provenance["input_sha256"]:
            raise ValueError("AU Mic input changed; use --refit-aumic")
        if full.sha256(template_path) != provenance["template_sha256"]:
            raise ValueError("AU Mic exclusion template changed; use --refit-aumic")
        if provenance["fit_windows_nm"] != [[592.05, 592.22]]:
            raise ValueError("AU Mic uses older fitting windows; use --refit-aumic")
        return table, provenance
    if plot_only:
        raise FileNotFoundError("AU Mic comparison has not been fitted yet")
    source.mkdir(parents=True, exist_ok=True)
    full.run_corrections(
        input_path,
        template_path,
        source,
        prepare_options={
            "wavelength_range_nm": (586.0, 594.0),
            "fit_windows_nm": ((592.05, 592.22),),
            "holdout_nm": (),
            "core_nm": ((588.95, 589.35), (589.55, 589.95)),
            "stellar_options": {
                "radial_velocity_kms": -4.42,
                "vsini_kms": 9.6,
                "mask_depth": 0.7,
                "mask_padding_kms": 3,
            },
            "comparison_protocol": AUMIC_PROTOCOL,
            "stellar_frame_from_metadata": True,
        },
        settings_overrides={
            "loss": "linear",
            "wavelength_shift_bounds": (-0.00001, 0.00001),
            "auto_segment": False,
            "fit_wavelength_polynomial": False,
            "fit_wavelength_shift": True,
        },
        molecfit_options={"WLC_N": 0},
    )
    full.collect_results(source)
    return load_products(source, AUMIC_PROTOCOL)


def crop_region(table, name):
    lower, upper = REGIONS[name][1]
    wave = np.asarray(table["wavelength_nm"])
    if wave.min() > lower or wave.max() < upper:
        raise ValueError(f"The saved spectrum does not cover the complete {name} window")
    crop = table[(wave >= lower) & (wave <= upper)].copy()
    if len(crop) < 2 or not np.any(crop["valid"]):
        raise ValueError(f"No usable pixels in {name}")
    return crop


def make_figure(crops, provenance, aumic=None):
    import matplotlib.pyplot as plt

    crops = dict(crops)
    if aumic is not None:
        crops["aumic"] = crop_region(aumic[0], "nad")
    mosaic = len(crops) > 1
    fig, axes = plt.subplots(
        1,
        len(crops),
        figsize=(13, 4.3) if mosaic else (11, 4.8),
        squeeze=False,
    )
    for index, (name, table) in enumerate(crops.items()):
        ax = axes.flat[index]
        wave = np.asarray(table["wavelength_nm"])
        # Stellar/core exclusions limit fitting and scores, not the displayed spectrum.
        observed_valid = np.asarray(table["valid"], dtype=bool) & np.isfinite(table["flux"])
        ax.plot(
            wave,
            np.where(observed_valid, table["flux"], np.nan),
            color="#009E73",
            lw=0.8,
            alpha=0.85,
            label="Uncorrected",
            zorder=1,
        )
        valid = observed_valid.copy()
        for engine in COLORS:
            valid &= np.isfinite(table[f"{engine}_corrected"])
            valid &= np.asarray(table[f"{engine}_transmission"]) > 0.01
        for engine, label in (("pymolfit", "PyMolFit"), ("molecfit", "Molecfit")):
            ax.plot(
                wave,
                np.where(valid, table[f"{engine}_corrected"], np.nan),
                color=COLORS[engine],
                ls="-" if engine == "pymolfit" else "--",
                lw=0.9,
                label=label,
                zorder=3,
            )
        _, limits = REGIONS["nad"]
        # Trim 0.5 nm from each display edge only.
        limits = (limits[0] + 0.5, limits[1] - 0.5)
        if name == "aumic":
            limits = (588.75, 590.12)
            ax.set_ylim(top=0.85)
        target = "AU Mic | ESPRESSO" if name == "aumic" else "Beta Pictoris | HARPS"
        date = (aumic[1] if name == "aumic" else provenance)["date_obs"][:10]
        ax.set(xlim=limits, title=f"{target} | {date}")
        ax.title.set_fontsize(13.7)
        ax.tick_params(axis="both", labelsize=11.7)
        ax.grid(alpha=0.15)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    for ax in axes.flat[len(crops) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Na I D doublet | Before and after correction",
        fontsize=15.7,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=12.55,
    )
    fig.supylabel("Flux [common scaled input units]", fontsize=13.7)
    fig.subplots_adjust(
        left=0.09 if mosaic else 0.115,
        right=0.965,
        bottom=0.19 if mosaic else 0.11,
        top=0.80 if mosaic else 0.82,
        hspace=0.34,
        wspace=0.26,
    )
    return fig


def save_regions(
    source,
    output,
    *,
    show,
    aumic_source=DEFAULT_AUMIC_SOURCE,
    aumic_input=DEFAULT_AUMIC_INPUT,
    aumic_template=DEFAULT_AUMIC_TEMPLATE,
    refit_aumic=False,
    plot_only=False,
):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output.suffix.lower() != ".png":
        raise ValueError("The figure output must be a PNG file")
    table, provenance = load_products(source)
    crops = {name: crop_region(table, name) for name in REGIONS}
    aumic = ensure_aumic(
        aumic_source, aumic_input, aumic_template, refit=refit_aumic, plot_only=plot_only
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure(crops, provenance, aumic)
    try:
        fig.savefig(output, dpi=180)
        print(f"Included catalogue species: {', '.join(provenance['species_policy']['included'])}")
        print(f"Saved {output}.", flush=True)
        if show:
            plt.show()
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE, help="Full-spectrum saved fit products"
    )
    parser.add_argument("--output", type=Path, default=HERE / "regions_mosaic.png")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aumic-source", type=Path, default=DEFAULT_AUMIC_SOURCE)
    parser.add_argument("--aumic-input", type=Path, default=DEFAULT_AUMIC_INPUT)
    parser.add_argument("--aumic-template", type=Path, default=DEFAULT_AUMIC_TEMPLATE)
    parser.add_argument("--refit-aumic", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        save_regions(
            args.source.expanduser().resolve(),
            args.output.expanduser().resolve(),
            show=args.show,
            aumic_source=args.aumic_source,
            aumic_input=args.aumic_input,
            aumic_template=args.aumic_template,
            refit_aumic=args.refit_aumic,
            plot_only=args.plot_only,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(
            f"{error}. Generate matching products with paper_validation/beta_pic_full/compare_full.py first."
        )


if __name__ == "__main__":
    main()
