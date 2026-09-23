"""CCD component definitions are parsed, bundled and cached."""
import pytest

from qpocket.io import components

ALA_CIF = """\
data_ALA
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
ALA N   CA  SING N N 1
ALA CA  C   SING N N 2
ALA C   O   DOUB N N 3
ALA CB  HB1 SING N N 4
#
"""


def test_parse_reads_orders_and_pairs():
    bonds = components.parse_component_cif(ALA_CIF, "ALA")
    assert bonds[frozenset({"N", "CA"})] == "SING"
    assert bonds[frozenset({"C", "O"})] == "DOUB"
    assert len(bonds) == 4


def test_parse_ignores_rows_of_other_components():
    mixed = ALA_CIF + "GLY N   CA  SING N N 1\n"
    assert len(components.parse_component_cif(mixed, "ALA")) == 4


def test_standard_residues_are_bundled_and_need_no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("must not reach the network for a bundled component")

    monkeypatch.setattr(components, "_download", refuse)
    components.component_bonds.cache_clear()
    for comp in ("ALA", "ARG", "TRP", "HIS", "HOH"):
        assert components.component_bonds(comp)


def test_unknown_component_names_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(components, "_download", lambda comp, path: False)
    components.component_bonds.cache_clear()
    with pytest.raises(components.MissingComponentError) as excinfo:
        components.component_bonds("ZZZ", cache_dir_override=tmp_path)
    assert excinfo.value.comp_id == "ZZZ"


def test_a_fetched_component_is_cached_on_disk(monkeypatch, tmp_path):
    calls = []

    def fake_download(comp, path):
        calls.append(comp)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ALA_CIF.replace("ALA", "LIG"), encoding="utf-8")
        return True

    monkeypatch.setattr(components, "_download", fake_download)
    components.component_bonds.cache_clear()
    components.component_bonds("LIG", cache_dir_override=tmp_path)
    components.component_bonds.cache_clear()
    components.component_bonds("LIG", cache_dir_override=tmp_path)
    assert calls == ["LIG"]
