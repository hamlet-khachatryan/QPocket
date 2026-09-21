"""The 3Dmol.js build the viewer runs on.

A vendored copy ships inside the package so the viewer works with no network
access, which matters on a locked-down machine or an offline compute node. The
CDN URL is kept as an alternative because inlining costs half a megabyte in every
rendered page.
"""

from __future__ import annotations

from pathlib import Path

#: Version vendored and pinned. Every 3Dmol API the viewer calls was checked
#: against this build; a different version needs rechecking.
THREEDMOL_VERSION = "2.4.2"

#: Immutable CDN URL for the same build.
THREEDMOL_CDN = (
    f"https://cdnjs.cloudflare.com/ajax/libs/3Dmol/{THREEDMOL_VERSION}/3Dmol-min.js"
)

#: sha256 of the vendored file, so a swapped asset is detectable.
THREEDMOL_SHA256 = "06b6d2fc7d418e8bef62a32cca44373ff21b2b787bd377613dd4039394a4e9ff"

_VENDOR = Path(__file__).with_name("vendor")


def threedmol_path() -> Path:
    """Path to the vendored 3Dmol build.

    Returns:
        Location of the minified JavaScript file.

    Raises:
        FileNotFoundError: If the package was installed without its vendored
            asset.
    """
    path = _VENDOR / f"3Dmol-{THREEDMOL_VERSION}.min.js"
    if not path.exists():
        raise FileNotFoundError(
            f"vendored 3Dmol build missing at {path}; reinstall the package or"
            f" download {THREEDMOL_CDN}"
        )
    return path


def threedmol_source() -> str:
    """Read the vendored 3Dmol source.

    Returns:
        The minified JavaScript.
    """
    return threedmol_path().read_text(encoding="utf-8")


def threedmol_script(offline: bool = True) -> str:
    """Build the script tag that loads 3Dmol into a page.

    Args:
        offline: Inline the vendored source, so the page needs no network. When
            ``False`` the page fetches the pinned CDN build instead, which keeps
            it around half a megabyte smaller.

    Returns:
        HTML to place in the page.
    """
    if not offline:
        return f'<script src="{THREEDMOL_CDN}"></script>'
    source = threedmol_source()
    # The minified build contains no "</script" sequence, so it can be inlined
    # verbatim; guard anyway, since a future version might not be so convenient.
    source = source.replace("</script", "<\\/script")
    return f"<script>{source}</script>"
