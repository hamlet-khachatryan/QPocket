"""Any docking binary can be declared in config and run from a task folder."""


from qpocket.screen_config import ScreenConfig, validate_screen_config

SPEC = {"executable": "/opt/smina", "output": "sdf", "score_property": "minimizedAffinity",
        "command": "{exe} -r {receptor} -l {ligand} -o {output} --center_x {center_x} --center_y {center_y} "
                   "--center_z {center_z} --size_x {size_x} --size_y {size_y} --size_z {size_z} "
                   "--exhaustiveness {exhaustiveness} --num_modes {num_modes} --cpu {cpus}"}


def _job():
    return {"protein_pdbqt": "r.pdbqt", "ligand_pdbqt": "l.pdbqt", "output_pdbqt": "poses/j.sdf",
            "center_x": 1, "center_y": 2, "center_z": 3, "size_x": 20, "size_y": 20, "size_z": 20,
            "exhaustiveness": 8, "num_modes": 9, "energy_range": 3.0, "engine": "smina"}


def test_a_custom_engine_can_be_the_engine():
    cfg = ScreenConfig()
    cfg.docking.custom_engines = {"smina": SPEC}
    cfg.docking.engine = "smina"
    validate_screen_config(cfg)


