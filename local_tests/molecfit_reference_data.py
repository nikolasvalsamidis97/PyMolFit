from __future__ import annotations

from pathlib import Path

from genmolfit.aer_data import (
    AER_CATALOG_FILENAME,
    AER_FILE_SPECS,
    AERCatalogArtifact,
    install_aer_catalog,
)


def stage_aer_molecfit_data(
    molecfit_data_root: str | Path,
    destination: str | Path,
    *,
    catalog: AERCatalogArtifact | None = None,
) -> tuple[Path, AERCatalogArtifact]:
    """Stage Molecfit data with GenMolFit's verified AER catalogue.

    The temporary tree reuses the installed Molecfit profiles and non-catalogue
    HITRAN files through symlinks. It never changes the Molecfit installation.
    """

    source_root = Path(molecfit_data_root).expanduser().resolve()
    if not (source_root / "profiles").is_dir() or not (source_root / "hitran").is_dir():
        raise FileNotFoundError(f"invalid Molecfit data root: {source_root}")
    artifact = catalog or install_aer_catalog(reuse_molecfit=False)

    staged_root = Path(destination).resolve()
    staged_root.mkdir(parents=True, exist_ok=False)
    for source in source_root.iterdir():
        if source.name == "hitran":
            continue
        (staged_root / source.name).symlink_to(source, target_is_directory=source.is_dir())

    staged_hitran = staged_root / "hitran"
    staged_hitran.mkdir()
    for source in (source_root / "hitran").iterdir():
        (staged_hitran / source.name).symlink_to(
            source,
            target_is_directory=source.is_dir(),
        )

    for filename in AER_FILE_SPECS:
        destination_path = staged_hitran / filename
        if destination_path.exists() or destination_path.is_symlink():
            destination_path.unlink()
        source_path = (
            artifact.catalog_path
            if filename == AER_CATALOG_FILENAME
            else artifact.extra_broadener_dir / filename
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination_path.symlink_to(source_path)

    return staged_root, artifact
