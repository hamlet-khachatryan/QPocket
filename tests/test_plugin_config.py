"""Plug-ins are chosen and tuned from config; bad names and options fail at load time."""

from tests.conftest import make_config


def test_the_chain_reads_its_plugins_from_config():
    from qpocket.pipelines.structure import structure_params

    p = structure_params(make_config())
    assert p.contacts == ("prolif",) and set(p.contact_params) == {"prolif"}


