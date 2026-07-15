from types import SimpleNamespace

from pymolfit.aer_data import AER_CATALOG_FILENAME, AER_FILE_SPECS
from local_tests.molecfit_reference_data import stage_aer_molecfit_data


def test_stage_aer_molecfit_data_overlays_catalog_without_mutating_install(tmp_path):
    source = tmp_path / "molecfit-data"
    (source / "profiles").mkdir(parents=True)
    (source / "profiles" / "profile.dat").write_text("profile")
    (source / "hitran").mkdir()
    (source / "hitran" / "legacy.dat").write_text("legacy")
    (source / "hitran" / "o2_h2o_brd_param").write_text("old")

    catalogue = tmp_path / "catalogue"
    catalogue.mkdir()
    for filename in AER_FILE_SPECS:
        (catalogue / filename).write_text(f"new {filename}")
    artifact = SimpleNamespace(
        catalog_path=catalogue / AER_CATALOG_FILENAME,
        extra_broadener_dir=catalogue,
    )

    staged, returned = stage_aer_molecfit_data(
        source,
        tmp_path / "staged",
        catalog=artifact,
    )

    assert returned is artifact
    assert (staged / "profiles" / "profile.dat").read_text() == "profile"
    assert (staged / "hitran" / "legacy.dat").read_text() == "legacy"
    assert (staged / "hitran" / "o2_h2o_brd_param").read_text() == "new o2_h2o_brd_param"
    assert (staged / "hitran" / AER_CATALOG_FILENAME).resolve() == artifact.catalog_path
    assert (source / "hitran" / "o2_h2o_brd_param").read_text() == "old"
