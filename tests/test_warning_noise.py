"""A run should not print third-party notices its user cannot act on."""
import subprocess
import sys


def test_importing_prolif_through_the_helper_is_quiet():
    """MDAnalysis announces that one of its own modules moved; that is not ours."""
    done = subprocess.run(
        [sys.executable, "-c",
         "from qpocket.core.detect import load_prolif; load_prolif()"],
        capture_output=True, text=True, timeout=300,
    )
    assert "MDAnalysis.topology.tables" not in done.stderr
    assert done.returncode == 0


def test_other_warnings_still_reach_the_user():
    """Silencing one notice must not silence the rest."""
    done = subprocess.run(
        [sys.executable, "-c",
         "import warnings\n"
         "from qpocket.core.detect import load_prolif\n"
         "load_prolif()\n"
         "warnings.warn('a real problem', UserWarning)\n"],
        capture_output=True, text=True, timeout=300,
    )
    assert "a real problem" in done.stderr


def test_the_spectral_laplacian_warns_about_nothing():
    """np.power(where=) without out= leaves uninitialised memory, and says so."""
    done = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c",
         "import numpy as np\n"
         "from qpocket.core.spectral import fingerprint, fit_scaler\n"
         "rng = np.random.default_rng(0)\n"
         "d = rng.random((24, 24)); d = (d + d.T) / 2; np.fill_diagonal(d, 0)\n"
         "f = rng.random((24, 3))\n"
         "fingerprint(d, f, fit_scaler([f]), 8, 10.0)\n"],
        capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0, done.stderr
