"""The reference is built once and survives the trip to disk."""
import numpy as np
import pytest

from qpocket.core.types import RawStructure
from qpocket.io import layout
from qpocket.pipelines import reference


def test_load_reference_reads_what_build_wrote(tmp_path, monkeypatch):
    from qpocket.core.types import AlignmentContext, Reference, Structure

    ws = layout.create(str(tmp_path), "run")
    ref = Reference("7KG6", AlignmentContext("a", "A", "M", None, {}, {}, {}, False, False),
                    Structure("7KG6", "x.cif"), np.ones(3), None)
    layout.write_reference(ws, ref.to_json())
    back = reference.load_reference(ws)
    assert back.pdb_id == "7KG6" and np.array_equal(back.centroid, np.ones(3))


def test_a_reference_that_cannot_be_cleaned_stops_the_run(tmp_path, monkeypatch):
    from qpocket.core.types import AlignmentContext, Structure
    from qpocket.pipelines import structure

    ws = layout.create(str(tmp_path), "run")
    ctx = AlignmentContext("a", "A", "M", None, {}, {}, {}, False, False)
    monkeypatch.setattr(reference, "alignment_context",
                        lambda cfg, ws, src: (ctx, Structure("7KG6", "x", 0.0, "a.pdb")))
    monkeypatch.setattr(structure, "clean_file", lambda record, ws, allowed: None)
    from tests.conftest import make_config

    with pytest.raises(ValueError, match="could not be cleaned"):
        reference.build_reference(make_config(), ws, RawStructure("7KG6", tmp_path / "7KG6.cif"))


def test_a_missing_reference_file_is_reported(tmp_path):
    from tests.conftest import make_config

    ws = layout.create(str(tmp_path), "run")
    with pytest.raises(FileNotFoundError, match="7KG6"):
        reference.alignment_context(make_config(), ws, RawStructure("7KG6", ws.original / "7KG6.cif"))
