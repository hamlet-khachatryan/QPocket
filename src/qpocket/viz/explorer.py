"""Interactive explorer for a Q-Pocket run, opened by ``qpocket explore``.

Its landscape, distance, cluster and ligand logic lives in ``qpocket.viz``.
"""

# Carrying no inline dependency metadata is deliberate: marimo offers to run a
# notebook that has it inside a sandboxed venv, that prompt defaults to yes, and
# the venv cannot import qpocket, so the kernel dies before any cell runs.

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from functools import lru_cache
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    import plotly.express as px

    import qpocket.viz as qviz
    from qpocket.arms import sites as qsites
    from qpocket.viz import controls as qcontrols
    from qpocket.viz import pages as qpages
    from qpocket.viz.pages import site as qsite_page
    from qpocket.viz.pages import exploration as qexplore
    from qpocket.arms.sites import final as qfinal
    from qpocket.arms.screen import box as qbox
    from qpocket.arms.screen import crossdock as qcrossdock
    from qpocket.arms.screen import submit as qsubmit
    from qpocket.viz.pages import screening as qscreen

    return (Path, json, lru_cache, mo, np, pd, px, qbox, qcontrols, qcrossdock, qexplore, qfinal, qpages, qscreen,
            qsite_page, qsites, qsubmit, qviz)


@app.cell
def _(mo):
    mo.md(r"""
    # Q-Pocket
    Interface for Interactive Exploration of the Target’s Experimentally Evidenced Conformational Landscape
    """)
    return


@app.cell
def _(mo):
    try:
        _cli_runs = mo.cli_args().get("runs")
    except Exception:
        _cli_runs = None
    run_location = mo.ui.text(
        value=_cli_runs or "",
        placeholder="/path/to/run  or  /path/to/folder/of/runs",
        label="…or open a path",
        full_width=True,
    )
    return (run_location,)


@app.cell
def _(Path, mo, qviz, run_location):
    try:
        _nb_dir = mo.notebook_dir()
    except Exception:
        _nb_dir = Path.cwd()

    _typed = qviz.resolve_location(run_location.value)
    _found = qviz.find_runs(qviz.default_roots(_nb_dir))
    run_dirs = list(dict.fromkeys([*_typed.runs, *_found]))  

    run_options = {}
    for _p in run_dirs:
        _label = qviz.run_label(_p)
        run_options[str(_p) if _label in run_options else _label] = str(_p)

    _pool = _typed.runs or run_dirs
    _complete = [p for p in _pool if qviz.is_complete(p)]
    _default = _complete[0] if _complete else (_pool[0] if _pool else None)
    _default_label = next((k for k, v in run_options.items() if v == str(_default)), None)

    run_selector = mo.ui.dropdown(options=run_options, value=_default_label, label="**Run**")
    mo.vstack([
        mo.hstack([run_selector, run_location], widths=[1, 2], align="end"),
        mo.md(f"_{_typed.message}_") if _typed.message else mo.md(""),
    ])
    return (run_selector,)


@app.cell
def _(mo, qpages):
    # The tab bar. Its panels are empty: sections render in their own cells, wrapped by qpages.on,
    # and the CSS below shows one page, so switching tabs reruns nothing and controls keep their values.
    page_tabs = mo.ui.tabs({name: "" for name in qpages.PAGES})
    page_tabs
    return (page_tabs,)


@app.cell
def _(mo, page_tabs, qpages):
    mo.Html(qpages.page_css(page_tabs.value or qpages.PAGES[0]))
    return


@app.cell
def _(Path, mo, run_selector):
    mo.stop(run_selector.value is None, mo.md("No run selected / none found."))
    run_dir = Path(run_selector.value)
    return (run_dir,)


@app.cell
def _(get_sites_version, mo, pd, qpages, qsites, run_dir):
    # Bumped only by a confirmed site change, so previewing one never re-reads the table or resets the picker.
    _ = get_sites_version()
    _empty = pd.DataFrame(columns=["site_id", "n_structures", "n_holo", "n_apo", "n_cavfind_miss",
                                   "n_ligands", "lining_residues", "centre_x", "centre_y", "centre_z"])
    try:
        sites_df = qsites.site_table(run_dir)
    except qsites.SiteError:
        sites_df = _empty
    # A run made before cavity-defined sites has the ligand-linked site table.
    _legacy = len(sites_df) > 0 and "n_structures" not in sites_df.columns
    if _legacy:
        sites_df = _empty
    _labels = {f"{r.site_id} · {r.n_structures} structures · {r.n_holo} holo · {r.n_cavfind_miss} misses":
               r.site_id for r in sites_df.itertuples()}
    site_picker = mo.ui.dropdown(options=_labels, value=next(iter(_labels), None),
                                 label="**Binding site**")
    qpages.on("Site", mo.vstack([
        mo.md("## Binding sites\nSites built from CavFind cavities seen across the run's structures. "
              "Everything below reads the site picked here; ligands CavFind found no cavity for are in "
              "`sites/cavfind_misses.csv`."),
        mo.callout(mo.md("This run's sites were made before cavity-defined sites. Re-site it to explore "
                         f"them: `qpocket site regroup {run_dir} --cavity-overlap 0.5`, or **Regroup sites** "
                         "below."), kind="warn") if _legacy else mo.md(""),
        site_picker,
        mo.ui.table(sites_df[["site_id", "n_structures", "n_holo", "n_apo", "n_cavfind_miss", "n_ligands",
                              "lining_residues"]],
                    selection=None, pagination=True),
    ]))
    return site_picker, sites_df


@app.cell
def _(mo):
    # The last site step's outcome; setting it re-reads the files that step rewrote.
    get_site_steps, set_site_steps = mo.state(None)
    return get_site_steps, set_site_steps


@app.cell
def _(Path, get_site_steps, mo, pd, qpages, run_dir, site_picker):
    mo.stop(site_picker.value is None, qpages.on(("Exploration", "Screening"), mo.callout(mo.md(
        "Pick a site on the Site page. This run has none yet; regroup its sites there."), kind="info")))

    _ = get_site_steps()
    site_dir = run_dir / "sites" / site_picker.value
    meta = site_dir / "metadata"
    pockets_dir = site_dir / "pockets"

    def _read_csv(p, **kw):
        return pd.read_csv(p, **kw) if p.exists() else None

    master_df = _read_csv(meta / "master.csv")
    clustering_df = _read_csv(meta / "clustering_metadata.csv")
    util_df = _read_csv(meta / "binding_pocket_utilization.csv")


    actives_df = _read_csv(site_dir / "screening" / "actives.smi", sep="\t")

    # A site clustered on shape alone has no spectral matrix; its shape matrix stands in.
    dist_kind = "spectral"
    dist_path = meta / "spectral_distance_matrix.csv"
    if not dist_path.exists():
        dist_kind, dist_path = "shape", meta / "shape_distance_matrix.csv"
    dist_df = pd.read_csv(dist_path, index_col=0) if dist_path.exists() else None

    #per-structure metadata
    info = clustering_df.copy() if clustering_df is not None else pd.DataFrame()
    if master_df is not None:
        keep = [
            c
            for c in ["pdb_id", "rmsd_to_ref", "protein_state", "ligand_file", "aligned_file", "cleaned_file"]
            if c in master_df.columns
        ]
        info = info.merge(master_df[keep], on="pdb_id", how="left") if len(info) else master_df[keep].copy()
    if util_df is not None:
        info = info.merge(util_df, on="pdb_id", how="left")

    info["is_representative"] = info.get("is_representative", False)
    info["cluster_id"] = info.get("cluster_id", -1)
    info = info.sort_values(["cluster_id", "pdb_id"]).reset_index(drop=True)

    def cloud_path(pdb_id: str) -> Path:
        """The pocket's featurised cloud, or its bare cloud before Featurise."""
        _featurized = pockets_dir / str(pdb_id) / "featurized.csv"
        return _featurized if _featurized.exists() else _featurized.with_name("binding_site.csv")

    #available features
    FEATURE_COLS = []
    for _pid in info["pdb_id"]:
        _fp = pockets_dir / str(_pid) / "featurized.csv"
        if _fp.exists():
            _cols = pd.read_csv(_fp, nrows=1).columns.tolist()
            FEATURE_COLS = [c for c in _cols if c not in ("x", "y", "z")]
            break

    qpages.on("Exploration", mo.md(
        f"""
        `{site_dir}` — structures: **{len(info)}** · clusters:
        **{info['cluster_id'].nunique()}** · representatives:
        **{int(info['is_representative'].sum())}** · features:
        `{', '.join(FEATURE_COLS) or 'none found'}`
        """
    ))
    return FEATURE_COLS, actives_df, cloud_path, dist_df, dist_kind, info, site_dir


@app.cell
def _(mo):
    site_ligand_limit = mo.ui.number(start=1, stop=200, step=1, value=25, label="Ligands shown")
    return (site_ligand_limit,)


@app.cell
def _(Path, info, mo, qpages, qviz, run_dir, site_dir, site_ligand_limit, sites_df):
    # Some of the site's ligands (one per holo structure, at most the limit) over the frame
    # protein, every site's centre as a sphere (this one dark, the others faint).
    _centres = sites_df[["centre_x", "centre_y", "centre_z"]].to_numpy(dtype=float)
    _here = (sites_df["site_id"] == site_dir.name).to_numpy(dtype=float)
    _files = qviz.site_view_ligands(run_dir, site_dir.name, int(site_ligand_limit.value))
    # shortcut: the first aligned structure stands in for the frame; every one is superposed onto it.
    _frame = next((run_dir / f for f in info.get("aligned_file", [])
                   if isinstance(f, str) and f and (run_dir / f).exists()), None)
    _ramp = qviz.RampSpec("sequential", "Blues", ("other sites", "this site"))
    _domain = qviz.ColourDomain(0.15, 1.0, None)
    _stops = qviz.ramp_stops(_ramp.scale)
    _panel = {
        "id": site_dir.name, "title": f"{site_dir.name} · {len(_files)} ligands shown",
        "cloud": qviz.pseudo_atom_pdb(_centres, b=qviz.scale_to_percent(_here, _domain)),
        "cartoon": _frame.read_text(encoding="utf-8") if _frame else "",
        "side": "", "highlights": {}, "n_forced": 0,
        "ligand": "".join(qviz.ligand_pdb(f) for f in _files if Path(f).exists()),
        "vals": [round(float(v), 3) for v in _here], "n_points": len(_centres), "n_ctx": 0,
    }
    _html, _h = qviz.grid.grid_html(
        [_panel], bbox_pdb=qviz.pseudo_atom_pdb(qviz.bbox_corners(_centres, pad=10.0), resname="BOX", chain="Z"),
        n_cols=1, panel_px=420, sphere_radius=1.2, stops=_stops, value_label="this site",
        ramp=_ramp, domain=_domain, hover=False, cartoon_opacity=0.5, ligand_color="greenCarbon",
        observed_colour="#eb6834", potential_colour="#7b5cd6", offline=True)
    qpages.on("Site", mo.vstack([site_ligand_limit, mo.iframe(_html, height=f"{_h + 70}px")]))
    return


@app.cell
def _(get_site_steps, mo, qpages, qsites, run_dir, site_dir):
    _sid = site_dir.name
    site_steps = qsites.site_status(run_dir, _sid)
    _no_clouds = " (needs clouds)" if not site_steps["clouds"] else ""
    shape_button = mo.ui.run_button(label="Shape clustering" + _no_clouds, disabled=bool(_no_clouds))
    featurise_button = mo.ui.run_button(label="Featurise" + _no_clouds, disabled=bool(_no_clouds))
    try:  # the run's own setting by default, so the field never overrides it unasked
        _workers = int(qsites.load_run_config(run_dir).compute.workers)
    except Exception:
        _workers = 0
    featurise_workers = mo.ui.number(start=-1, stop=512, step=1, value=_workers,
                                     label="Workers (0 = half the cores, -1 = all)")
    # two_stage_site's own precondition: two featurised pockets, not every cloud.
    two_stage_button = mo.ui.run_button(
        label=f"Two-stage clustering ({site_steps['n_featurised']}/{site_steps['n_clouds']} featurised)",
        disabled=site_steps["n_featurised"] < 2)
    _done = " · ".join(f"{name.replace('_', ' ')} {'✓' if ok else '—'}"
                       for name, ok in site_steps.items() if isinstance(ok, bool))
    _commands = "\n".join([
        f"qpocket site shape {run_dir} --site {_sid}",
        f"qpocket site featurize {run_dir} --site {_sid}",
        f"qpocket site cluster {run_dir} --site {_sid}",
        "# or on SLURM (featurises one cloud per array task, then clusters):",
        f"qpocket slurm generate --config {run_dir / 'metadata' / 'config.yaml'} "
        f"--workdir {run_dir.parent} --site {_sid}",
    ])
    qpages.on("Exploration", mo.vstack([
        mo.md(f"### Steps for {_sid}\n{_done}"),
        mo.hstack([shape_button, featurise_workers, featurise_button, two_stage_button], justify="start",
                  align="end"),
        get_site_steps() or mo.md(""),
        mo.accordion({"Run them from a terminal instead": mo.plain_text(_commands)}),
    ]))
    return featurise_button, featurise_workers, shape_button, two_stage_button


@app.cell
def _(featurise_button, featurise_workers, mo, qsites, run_dir, set_site_steps, shape_button, site_dir,
      two_stage_button):
    mo.stop(not (shape_button.value or featurise_button.value or two_stage_button.value))
    import traceback as _traceback

    _name, _step = next((n, f) for n, f, b in [
        ("Shape clustering", qsites.shape_cluster_site, shape_button),
        ("Featurise", lambda r, s: qsites.featurize_site(r, s, workers=int(featurise_workers.value)),
         featurise_button),
        ("Two-stage clustering", qsites.two_stage_site, two_stage_button),
    ] if b.value)
    with mo.status.spinner(title=f"{_name} for {site_dir.name}…"):
        try:
            _step(run_dir, site_dir.name)
            _message = mo.md(f"_{_name} finished for {site_dir.name}._")
        except Exception as _e:
            _message = mo.vstack([
                mo.md(f"_{_name} failed: {type(_e).__name__}: {_e}_"),
                mo.accordion({"Traceback": mo.plain_text(_traceback.format_exc())}),
            ])
    set_site_steps(_message)
    return


@app.cell
def _(mo, qpages, qviz, run_dir):
    # This cell feeds the site table, so an odd config must not stop it.
    _p = qviz.site_parameters(run_dir)

    def _number(key, lo, hi, step, label):  # a saved value outside the control's range is clamped
        return mo.ui.number(start=lo, stop=hi, step=step, value=min(max(float(_p[key]), lo), hi), label=label)

    regroup_controls = mo.ui.dictionary({
        "cavity_overlap": _number("cavity_overlap", 0.05, 1.0, 0.05, "Cavity overlap (IoU)"),
        "ligand_fraction": _number("ligand_fraction", 0.05, 1.0, 0.05, "Ligand fraction"),
        "box_padding": _number("box_padding", 0.0, 10.0, 0.5, "Box padding (Å)"),
        "min_cavity_points": _number("min_cavity_points", 1, 2000, 1, "Min cavity points"),
        "min_structure_fraction": _number("min_structure_fraction", 0.0, 1.0, 0.01, "Min structure share"),
    })
    regroup_button = mo.ui.run_button(label="Regroup sites…")
    qpages.on("Site", mo.accordion({"Regroup sites": mo.vstack([
        mo.md("Rebuilds every site from the run's cavities with these parameters. You will see what "
              "would be lost and confirm first."),
        mo.hstack([*regroup_controls.elements.values(), regroup_button], justify="start", align="end", wrap=True),
        mo.plain_text(f"qpocket site regroup {run_dir} --cavity-overlap <x> --ligand-fraction <y> "
                      "--box-padding <z>"),
    ])}))
    return regroup_button, regroup_controls


@app.cell
def _(mo):
    # A site change waiting for Confirm: {"run", "label", "params"} for a regroup, {"run", "label", "overrides"}
    # for an edit; the outcome of the last one; and a counter a confirmed change bumps to re-read the sites.
    get_pending_site_change, set_pending_site_change = mo.state(None)
    get_site_change_message, set_site_change_message = mo.state("")
    get_sites_version, set_sites_version = mo.state(0)
    return (get_pending_site_change, get_site_change_message, get_sites_version, set_pending_site_change,
            set_site_change_message, set_sites_version)


@app.cell
def _(mo, qpages, qsite_page, run_dir, sites_df):
    edit_sites = mo.ui.multiselect(options=list(sites_df["site_id"]), label="Sites")
    merge_button = mo.ui.run_button(label="Merge…")
    drop_button = mo.ui.run_button(label="Drop…")
    split_button = mo.ui.run_button(label="Split…")
    _overrides = qsite_page.override_table(run_dir)
    undo_pick = mo.ui.dropdown(options={f"#{r['#']} {r.operation} {r.sites}": int(r["#"])
                                        for _, r in _overrides.iterrows()}, label="Undo")
    undo_button = mo.ui.run_button(label="Undo…", disabled=_overrides.empty)
    qpages.on("Site", mo.vstack([
        mo.md("### Edit sites\nMerge two or more sites, drop one (its ligands become misses), or split one "
              "into its strict cavity groups. Edits are kept in the run's config and re-applied after "
              "every regroup."),
        mo.hstack([edit_sites, merge_button, drop_button, split_button], justify="start", align="end"),
        mo.md("### Overrides"),
        mo.ui.table(_overrides, selection=None) if len(_overrides) else mo.md("_No manual edits._"),
        mo.hstack([undo_pick, undo_button], justify="start", align="end") if len(_overrides) else mo.md(""),
    ]))
    return drop_button, edit_sites, merge_button, split_button, undo_button, undo_pick


@app.cell
def _(drop_button, edit_sites, merge_button, mo, qpages, qsites, regroup_button, regroup_controls, run_dir,
      set_pending_site_change, split_button, undo_button, undo_pick):
    mo.stop(not any(b.value for b in (regroup_button, merge_button, drop_button, split_button, undo_button)))
    _picked = list(edit_sites.value)
    _current = [{k: v for k, v in o.items() if k not in ("status", "site_ids", "parts")}
                for o in qsites.list_overrides(run_dir)]
    try:
        if regroup_button.value:
            _params = {k: float(v) for k, v in regroup_controls.value.items()}
            _change = {"label": "regroup (" + ", ".join(f"{k} {v:g}" for k, v in _params.items()) + ")",
                       "params": _params}
        elif merge_button.value:
            _change = {"label": f"merge {' + '.join(_picked)}",
                       "overrides": _current + [qsites.merge_override(run_dir, _picked)]}
        elif undo_button.value:
            if undo_pick.value is None:
                raise qsites.SiteError("pick the override to undo")
            _change = {"label": f"undo override #{undo_pick.value}",
                       "overrides": [o for i, o in enumerate(_current) if i != undo_pick.value]}
        else:
            if len(_picked) != 1:
                raise qsites.SiteError("pick exactly one site")
            _make = qsites.drop_override if drop_button.value else qsites.split_override
            # A split on an older grouping works its strict groups out again, which takes a while.
            with mo.status.spinner(title=f"Reading {_picked[0]}'s cavity groups…"):
                _override = _make(run_dir, _picked[0])
            _change = {"label": f"{'drop' if drop_button.value else 'split'} {_picked[0]}",
                       "overrides": _current + [_override]}
        set_pending_site_change({**_change, "run": str(run_dir)})
        _out = mo.md("")
    except Exception as _e:  # a site error is the user's to fix; anything else must not leave an old change pending
        set_pending_site_change(None)
        _out = mo.callout(mo.md(str(_e) if isinstance(_e, qsites.SiteError) else f"{type(_e).__name__}: {_e}"),
                          kind="warn")
    qpages.on("Site", _out)
    return


@app.cell
def _(get_pending_site_change, mo, qpages, qsite_page, qsites, run_dir):
    # A change requested on another run is never shown here, so it cannot be confirmed against this one.
    pending_site_change = get_pending_site_change()
    if pending_site_change is not None and pending_site_change.get("run") != str(run_dir):
        pending_site_change = None
    confirm_site_change = mo.ui.run_button(label="Confirm", kind="danger")
    cancel_site_change = mo.ui.run_button(label="Cancel")
    if pending_site_change is None:
        _out = mo.md("")
    else:
        with mo.status.spinner(title="Working out what would change…"):
            try:
                _p = qsites.preview(run_dir, overrides=pending_site_change.get("overrides"),
                                    **pending_site_change.get("params", {}))
                _counts = f"{_p.n_before} sites now, {_p.n_after} after ({_p.n_new} new or changed)."
                _with = {sid: items for sid, items in _p.losses.items() if items}
                _what = (f"{_counts} {len(_p.losses)} current site(s) change; these lose results:\n\n"
                         f"{qsite_page.loss_markdown(_with)}" if _with else
                         f"{_counts} {len(_p.losses)} current site(s) change; none holds results."
                         if _p.losses else f"{_counts} No current site's members change.")
            except Exception as _e:
                _what = f"Cannot preview this change: {type(_e).__name__}: {_e}"
        _out = mo.callout(mo.vstack([mo.md(f"**Pending: {pending_site_change['label']}.** {_what}"),
                                     mo.hstack([confirm_site_change, cancel_site_change], justify="start")]),
                          kind="warn")
    qpages.on("Site", _out)
    return cancel_site_change, confirm_site_change, pending_site_change


@app.cell
def _(cancel_site_change, confirm_site_change, mo, pending_site_change, qsites, run_dir,
      set_pending_site_change, set_site_change_message, set_sites_version):
    # Outputs nothing the site table reads: the table re-reads on the version bump, and the outcome is kept in
    # state, so this cell rerunning (it does, whenever the pending change is cleared) wipes neither.
    if cancel_site_change.value:
        set_pending_site_change(None)
        set_site_change_message("")
    elif confirm_site_change.value and pending_site_change is not None \
            and pending_site_change.get("run") == str(run_dir):
        with mo.status.spinner(title=f"Applying: {pending_site_change['label']}…"):
            try:
                if "params" in pending_site_change:
                    _table = qsites.regroup_sites(run_dir, **pending_site_change["params"])
                else:
                    _table = qsites.apply_site_overrides(run_dir, pending_site_change["overrides"])
                _message = f"Done: {pending_site_change['label']}; the run has {len(_table)} sites."
                _kind = "success"
            except Exception as _e:
                _message = f"{pending_site_change['label']} failed: {type(_e).__name__}: {_e}"
                _kind = "danger"
        set_site_change_message((str(run_dir), _kind, _message))
        set_pending_site_change(None)
        set_sites_version(lambda v: v + 1)
    return


@app.cell
def _(get_site_change_message, mo, qpages, run_dir):
    _last = get_site_change_message()
    qpages.on("Site", mo.callout(mo.md(_last[2]), kind=_last[1])
              if _last and _last[0] == str(run_dir) else mo.md(""))
    return


@app.cell
def _(mo, qpages, qsite_page, run_dir, site_picker, sites_df):
    qpages.on("Site", mo.vstack([
        mo.md("### Site overview"),
        qsite_page.overview_figure(run_dir, sites_df, selected=site_picker.value) if len(sites_df)
        else mo.md("_No sites to draw._"),
    ]))
    return


@app.cell
def _(lru_cache, pd):
    import os

    @lru_cache(maxsize=512)
    def _read_pocket(path: str, mtime: float) -> pd.DataFrame:
        return pd.read_csv(path)

    def load_pocket(path: str) -> pd.DataFrame:
        # Keyed on mtime too, so a cloud Featurise rewrites is read afresh.
        return _read_pocket(path, os.path.getmtime(path))

    return (load_pocket,)


@app.cell
def _(qviz, site_dir):
    # Read once, in a cell of its own: both the 3D panels and the fingerprint
    # section below depend on this, and marimo resolves cells by dataflow, so a
    # shared upstream cell is what keeps them from forming a cycle.
    run_interactions = qviz.interactions.load_run_interactions(site_dir)
    return (run_interactions,)


@app.cell
def _(mo, qpages):
    qpages.on("Exploration", mo.md("""
    ## Clustering
    Pick a clustering method and adjust the controls to select the best parametrization. **Save** updates the run's clustering while keeping the old one in `metadata/clustering_history/`.
    """))
    return


@app.cell
def _(mo, qpages, qviz, site_dir):
    from qpocket.arms import recluster as _recluster

    cluster_matrices = qviz.two_stage.load_matrices(site_dir)
    cluster_settings = _recluster.load_settings(site_dir)

    cluster_method = mo.ui.dropdown(
        options={label: key for key, label in qviz.two_stage.METHODS.items()},
        value=qviz.two_stage.METHODS["two_stage"],
        label="**Method**",
        full_width=True,
    )
    (qpages.on("Exploration", mo.hstack([cluster_method], justify="start", gap=2) if cluster_matrices is not None
     else mo.md("_This site has no distance matrices yet; run Shape clustering above._")))
    return cluster_matrices, cluster_method, cluster_settings


@app.cell
def _(cluster_settings, mo):
    # Every cut is a site percentile: the cut sits at that quantile of the site's pairwise distances.
    def _slider(value, label):
        return mo.ui.slider(start=0.5, stop=99.5, step=0.5, value=min(max(float(value), 0.5), 99.5),
                            label=label, show_value=True, debounce=True)

    damping = mo.ui.slider(start=0.5, stop=0.98, step=0.01, value=0.9,
                           label="Damping", show_value=True)
    combined_cut = _slider(cluster_settings.single.percentile, "Cut (percentile)")
    shape_cut = _slider(cluster_settings.shape.percentile, "Shape cut (percentile)")
    chem_cut = _slider(cluster_settings.chem.percentile, "Chemistry cut (percentile)")
    min_split = mo.ui.number(start=2, stop=100, value=int(cluster_settings.min_cluster_size),
                             label="Split groups of at least")
    return chem_cut, combined_cut, damping, min_split, shape_cut


@app.cell
def _(chem_cut, cluster_matrices, cluster_method, combined_cut, damping, min_split, mo, qpages, shape_cut):
    # Only the knobs the chosen method uses are on screen.
    (qpages.on("Exploration", mo.md("") if cluster_matrices is None else
     mo.vstack([mo.md("_Shape-only clustering — featurise this site for two-stage clustering._"),
                shape_cut]) if cluster_matrices.chem is None else {
        "affinity": mo.hstack([damping], justify="start", gap=2),
        "hierarchical": mo.hstack([combined_cut], justify="start", gap=2),
        "two_stage": mo.hstack([shape_cut, chem_cut, min_split], justify="start", gap=2),
    }[cluster_method.value]))
    return


@app.cell
def _(
    chem_cut,
    cluster_matrices,
    cluster_method,
    cluster_settings,
    combined_cut,
    damping,
    min_split,
    qviz,
    shape_cut,
):
    # The preview runs the pipeline's own clustering, so what is shown here and
    # what Apply writes cannot disagree.
    pending_settings, pending = None, None
    if cluster_matrices is not None:
        pending_settings = qviz.two_stage.shape_only_settings(
            cluster_settings, shape_cut.value) if cluster_matrices.chem is None else qviz.two_stage.settings_for(
            cluster_settings, cluster_method.value,
            combined_cut=combined_cut.value, shape_cut=shape_cut.value,
            chem_cut=chem_cut.value, damping=damping.value,
            min_cluster_size=int(min_split.value))
        pending = qviz.two_stage.preview(cluster_matrices, pending_settings)
    return pending, pending_settings


@app.cell
def _(chem_cut, cluster_matrices, cluster_method, cluster_settings, combined_cut, info, load_pocket, mo, pending, qpages, qviz, shape_cut, site_dir):
    # The tree is drawn only down to its cut, one leaf per cluster; the table beside it reads the same
    # preview, and both share the colours every view below uses.
    def _cloud(pdb_id):
        _path = site_dir / "pockets" / str(pdb_id) / "binding_site.csv"
        return load_pocket(str(_path))[["x", "y", "z"]].to_numpy(float) if _path.exists() else None

    cluster_table, tree_view, preview_colours, tree_matrices, tree_preview = None, mo.md(""), {}, None, None
    name_colours = {}
    if cluster_matrices is not None and pending is not None and cluster_method.value != "affinity":
        _single = cluster_method.value == "hierarchical" and cluster_matrices.chem is not None
        tree_matrices = (cluster_matrices._replace(shape=cluster_matrices.combined, chem=None) if _single
                         else cluster_matrices)
        tree_preview = (pending._replace(shape_labels=pending.labels, frame=pending.frame.assign(
            shape_cluster_id=pending.frame["cluster_id"], chem_cluster_id=0)) if _single else pending)
        _min = int(cluster_settings.min_cluster_size)
        preview_colours = qviz.style.cluster_colours(tree_preview.frame["shape_cluster_id"],
                                                     tree_preview.frame["chem_cluster_id"], _min)
        _chem = None if (_single or tree_matrices.chem is None) else chem_cut.value
        _shape = combined_cut.value if _single else shape_cut.value
        tree_view = qviz.two_stage.collapsed_figure(tree_matrices, tree_preview, _shape, _chem,
                                                    cluster_settings.shape.linkage, preview_colours)
        _info = tree_preview.frame.merge(info[["pdb_id", "protein_state"]].astype({"pdb_id": str}),
                                         on="pdb_id", how="left")
        _clouds = {pid: c for pid in tree_preview.frame["pdb_id"] if (c := _cloud(pid)) is not None}
        cluster_table = qviz.clusters.cluster_table(_info, tree_matrices, _clouds, _min)
        _f = tree_preview.frame
        for (_s, _c), _colour in preview_colours.items():
            name_colours.setdefault(qviz.clusters.cluster_name(_s), qviz.two_stage._shape_colour(
                preview_colours, _f, _s))
            name_colours[qviz.clusters.cluster_name(_s, _c)] = _colour
        _hued = sum(1 for _s in set(_f["shape_cluster_id"])
                    if qviz.two_stage._shape_colour(preview_colours, _f, _s) != qviz.style.GREY)
        if _hued > len(qviz.style.HUES):
            tree_view = mo.vstack([tree_view, mo.md(
                f"_{_hued} coloured clusters share {len(qviz.style.HUES)} hues: colours repeat every "
                f"{len(qviz.style.HUES)} clusters (names are unique)._")])
    elif cluster_method.value == "affinity" and pending is not None:
        tree_view = mo.md("_Affinity propagation builds no tree; see Cluster distances below._")
    qpages.on("Exploration", tree_view)
    return cluster_table, name_colours, preview_colours, tree_matrices, tree_preview


@app.cell
def _(cluster_table, mo, name_colours, qpages):
    def _fmt(v, pct=False):
        if v != v:  # NaN
            return "—"
        return f"{v * 100:.0f}%" if pct else f"{'▇' * max(1, round(v / 10))} p{v:.0f}"

    cluster_pick = None
    if cluster_table is not None and len(cluster_table):
        _shown = cluster_table.assign(
            holo=cluster_table["holo"].map(lambda v: _fmt(v, True)),
            shape_within=cluster_table["shape_within"].map(_fmt),
            chem_within=cluster_table["chem_within"].map(_fmt),
            shared_volume=cluster_table["shared_volume"].map(lambda v: _fmt(v, True)),
        ).rename(columns={"shape_within": "shape within", "chem_within": "chem within",
                          "shared_volume": "shared volume"})
        _shown["cluster"] = [("    ↳ " if lvl == "chem" else "■ ") + c
                             for c, lvl in zip(_shown["cluster"], _shown["level"])]

        def _style(_row, column, value):  # each cluster in its own colour, as in the tree
            _name = str(value).replace("■ ", "").replace("    ↳ ", "")
            return ({"color": name_colours.get(_name, "#6b737c"), "fontWeight": 600}
                    if column == "cluster" else {})

        cluster_pick = mo.ui.table(_shown.drop(columns=["level", "parent"]), selection="single",
                                   pagination=True, page_size=12, label="Clusters (select one)",
                                   style_cell=_style)
    qpages.on("Exploration", cluster_pick if cluster_pick is not None else mo.md(""))
    return (cluster_pick,)


@app.cell
def _(cluster_pick, cluster_settings, mo, qpages, qviz, tree_matrices, tree_preview):
    # The selected cluster's own tree, members on hover only.
    _view = mo.md("")
    if cluster_pick is not None and tree_preview is not None and len(cluster_pick.value):
        _name = str(cluster_pick.value.iloc[0]["cluster"]).replace("■ ", "").replace("    ↳ ", "")
        _frame = tree_preview.frame
        if _name != "small":
            _shape = int(_name.split("·")[0][1:]) - 1
            _rows = _frame[_frame["shape_cluster_id"] == _shape]
            _kind = "shape"
            if "·c" in _name:
                _rows = _rows[_rows["chem_cluster_id"] == int(_name.split("·c")[1]) - 1]
                _kind = "chem"
            _view = mo.vstack([mo.md(f"**{_name}** · {len(_rows)} pockets"),
                               qviz.two_stage.subtree_figure(tree_matrices, list(_rows["pdb_id"]), _kind,
                                                             cluster_settings.shape.linkage)])
    qpages.on("Exploration", _view)
    return


@app.cell
def _(info, mo, pending, qpages):
    def _groups(frame, column):
        return {frozenset(g["pdb_id"].astype(str)) for _, g in frame.groupby(column)}

    _facts = []
    if pending is not None:
        _facts.append(f"**{pending.n_clusters} clusters** in the preview")
        _same = "cluster_id" in info and set(info["cluster_id"].unique()) != {-1} and \
            _groups(pending.frame, "cluster_id") == _groups(info, "cluster_id")
        _facts.append("same as the applied clustering" if _same
                      else "Preview — apply to update the page")
    qpages.on("Exploration", mo.md(" · ".join(_facts) if _facts else ""))
    return


@app.cell
def _(mo, pending_settings, qpages):
    apply_clustering = mo.ui.run_button(
        label="Apply to this run", disabled=pending_settings is None)
    qpages.on("Exploration", mo.vstack([
        apply_clustering,
        mo.md("_Writes `metadata/clustering_metadata.csv` and `clustering_params.json`; "
              "the previous pair moves to `metadata/clustering_history/`._"),
    ]))
    return (apply_clustering,)


@app.cell
def _(apply_clustering, mo, pending_settings, qpages, site_dir):
    # Every section below reads the file this writes, so the button is what makes
    # them refresh.
    applied_at = None
    if apply_clustering.value and pending_settings is not None:
        from qpocket.arms import recluster as _recluster_apply

        _saved = _recluster_apply.run(site_dir, pending_settings)
        applied_at = str(_saved)
    qpages.on("Exploration", mo.md(f"Applied; `{applied_at}` is now this run's clustering."
          if applied_at else ""))
    return (applied_at,)


@app.cell
def _(applied_at, get_rep_version, info, mo, pd, qpages, site_dir):
    # ---- The grouping every view below follows -------------------------------
    # The clustering section is the decision point: it writes
    # metadata/clustering_metadata.csv, and this reads that file back. Depending
    # on applied_at is what makes these views refresh after Apply.
    _ = (applied_at, get_rep_version())
    active_info = info.copy()
    _stored_path = site_dir / "metadata" / "clustering_metadata.csv"
    if _stored_path.exists():
        _saved = pd.read_csv(_stored_path)
        _cols = [c for c in ["cluster_id", "shape_cluster_id", "chem_cluster_id",
                             "is_representative", "dist_to_exemplar"]
                 if c in _saved.columns]
        active_info = (
            active_info.drop(columns=[c for c in _cols if c in active_info.columns])
            .assign(pdb_id=lambda d: d["pdb_id"].astype(str))
            .merge(_saved[["pdb_id", *_cols]].assign(
                pdb_id=lambda d: d["pdb_id"].astype(str)), on="pdb_id", how="left")
            .sort_values(["cluster_id", "pdb_id"])
            .reset_index(drop=True)
        )
    qpages.on("Exploration", mo.md(
        f"**{active_info['cluster_id'].nunique()} pocket conformations** across "
        f"{len(active_info)} structures"
        + f" · {int(active_info['is_representative'].sum())} representatives"
    ))
    return (active_info,)


@app.cell
def _(mo):
    # Bumped by a representative swap, so every view re-reads the clustering; the swap's outcome.
    get_rep_version, set_rep_version = mo.state(0)
    get_rep_message, set_rep_message = mo.state(None)
    get_rep_choice, set_rep_choice = mo.state(None)  # the cluster last swapped, still picked afterwards
    return get_rep_choice, get_rep_message, get_rep_version, set_rep_choice, set_rep_message, set_rep_version


@app.cell
def _(active_info, get_rep_choice, mo, qviz):
    # cluster_id -1 marks a site that is not clustered yet: nothing to pick.
    _clustered = active_info[active_info["cluster_id"] != -1] if "cluster_id" in active_info else active_info.iloc[:0]
    _names = qviz.clusters.cluster_labels_of(_clustered) if len(_clustered) else {}
    _sizes = _clustered["cluster_id"].value_counts() if len(_names) else {}
    _options = {f"{_names[int(c)]} ({_sizes[c]})": int(c)
                for c in sorted(_sizes.index, key=lambda c: (-_sizes[c], c))} if len(_names) else {}
    _keep = next((label for label, c in _options.items() if c == get_rep_choice()), None)
    rep_cluster = mo.ui.dropdown(options=_options, value=_keep, label="Cluster")
    return (rep_cluster,)


@app.cell
def _(active_info, get_rep_message, mo, qexplore, qpages, rep_cluster, site_dir):
    rep_member = mo.ui.dropdown(
        options=qexplore.representative_options(active_info, rep_cluster.value) if rep_cluster.value is not None
        else {}, label="Representative")
    set_rep_button = mo.ui.run_button(label="Set representative", disabled=rep_cluster.value is None)
    qpages.on("Exploration", mo.vstack([
        mo.md("### Representatives\nEach cluster's medoid by default. Swap one here; cross-docking and Finalise "
              "use it. Applying a new clustering starts from medoids again."),
        mo.hstack([rep_cluster, rep_member, set_rep_button], justify="start", align="end"),
        mo.md(qpages.shown(get_rep_message(), qpages.clustering_stamp(site_dir))),
    ]))
    return rep_member, set_rep_button


@app.cell
def _(mo, qpages, qsites, rep_cluster, rep_member, run_dir, set_rep_button, set_rep_choice, set_rep_message,
      set_rep_version, site_dir):
    mo.stop(not set_rep_button.value)
    try:
        if rep_member.value is None:
            raise qsites.SiteError("pick the member to make representative")
        qsites.set_representative(run_dir, site_dir.name, rep_cluster.value, rep_member.value)
        _text = f"_{rep_member.value} is now the representative of its cluster._"
        set_rep_choice(rep_cluster.value)
        set_rep_version(lambda v: v + 1)
    except qsites.SiteError as _e:
        _text = f"_{_e}_"
    # Kept with the clustering it is about, so Apply or another site hides it.
    set_rep_message(qpages.stamped(_text, qpages.clustering_stamp(site_dir)))
    return


@app.cell
def _(mo, qpages):
    qpages.on("Exploration", mo.md("""
    ## Structures
    `dist_to_exemplar` — distance to the cluster representative (large = outlier) ·
    `utilization` — share of the reference site covered.
    """))
    return


@app.cell
def _(active_info, mo, qpages):
    _cols = ["pdb_id", "cluster_id"] + [
        c
        for c in [
            "is_representative", "dist_to_exemplar", "utilization",
            "rmsd_to_ref", "protein_state",
        ]
        if c in active_info.columns
    ]
    qpages.on("Exploration", mo.ui.table(active_info[_cols], selection=None, pagination=True))
    return


@app.cell
def _(dist_kind, mo, qpages):
    qpages.on("Exploration", mo.md(f"""
    ## Pocket landscape
    One point per structure, placed by {dist_kind} distance alone. **PCA** is the honest
    default; UMAP and t-SNE separate groups, but their gaps are not distances. Lasso
    points to open their cluster below.
    """))
    return


@app.cell
def _(mo):
    embed_method = mo.ui.dropdown(
        options=["PCA (classical MDS)", "UMAP", "t-SNE"],
        value="UMAP",
        label="Layout",
    )
    return (embed_method,)


@app.cell
def _(active_info, dist_df, embed_method, mo, qcontrols, qpages, qviz):
    # A cluster colour, or a number on a diverging scale; holo/apo is in every hover.
    _color_opts = [
        c
        for c in ["cluster_id", "utilization", "rmsd_to_ref"]
        if c in active_info.columns
    ]

    color_by = mo.ui.dropdown(
        options=_color_opts,
        value="cluster_id" if "cluster_id" in _color_opts else (_color_opts[0] if _color_opts else None),
        label="Colour by",
    )

    _n = 0 if dist_df is None else len(dist_df)
    n_neighbors = mo.ui.slider(
        start=2, stop=max(2, _n - 1), value=min(15, max(2, _n - 1)), label="n_neighbors"
    )
    min_dist = mo.ui.slider(start=0.0, stop=0.99, step=0.01, value=0.1, label="min_dist")
    _pmax = qviz.max_perplexity(_n)
    perplexity = mo.ui.slider(
        start=2.0, stop=round(_pmax, 1), step=0.5,
        value=round(min(15.0, _pmax), 1), label="perplexity",
    )

    # The method and what it is coloured by stay out; its knobs fold away.
    qpages.on("Exploration", qcontrols.panel(
        primary=[embed_method, color_by],
        groups={embed_method.value: {"UMAP": [n_neighbors, min_dist],
                                     "t-SNE": [perplexity]}.get(embed_method.value, [])},
        label="Layout options",
    ))
    return color_by, min_dist, n_neighbors, perplexity


@app.cell
def _(active_info, dist_df, embed_method, min_dist, mo, n_neighbors, pd, perplexity, qpages, qviz):
    mo.stop(dist_df is None, qpages.on("Exploration", mo.md("No distance matrix for this site yet; run Shape clustering above.")))

    _ids = list(dist_df.index)
    try:
        layout = qviz.embed_distances(
            dist_df.loc[_ids, _ids].to_numpy(dtype=float),
            method=embed_method.value,
            n_neighbors=int(n_neighbors.value),
            min_dist=float(min_dist.value),
            perplexity=float(perplexity.value),
        )
    except ValueError as _exc:
        mo.stop(True, qpages.on("Exploration", mo.md(f"Cannot lay this run out: {_exc}")))

    emb_df = pd.DataFrame(
        {"pdb_id": _ids, "X": layout.coords[:, 0], "Y": layout.coords[:, 1]}
    ).merge(active_info, on="pdb_id", how="left")
    return emb_df, layout


@app.cell
def _(active_info, cluster_settings, color_by, dist_kind, emb_df, layout, mo, qpages, qviz):
    # Small points, no printed labels: the pocket, its cluster and holo/apo show on hover.
    _cb = color_by.value
    _coords = emb_df.rename(columns={"X": "x", "Y": "y"})
    _shape_ids, _chem_ids = qviz.clusters.display_ids(_coords)
    _coords = _coords.assign(shape_cluster_id=_shape_ids, chem_cluster_id=_chem_ids,
                             protein_state=_coords["protein_state"] if "protein_state" in _coords else "")
    _colours = qviz.style.cluster_colours(_coords["shape_cluster_id"], _coords["chem_cluster_id"],
                                          int(cluster_settings.min_cluster_size))
    _fig = qviz.clusters.landscape_figure(_coords, "cluster" if _cb == "cluster_id" else _cb, _colours)
    _fig.update_layout(title=f"{layout.method} of {dist_kind} distances · coloured by "
                             f"{'cluster' if _cb == 'cluster_id' else _cb}",
                       xaxis_title=layout.axis_labels[0], yaxis_title=layout.axis_labels[1])
    embed_plot = mo.ui.plotly(_fig)
    qpages.on("Exploration", mo.vstack([embed_plot, mo.md(f"_{layout.note}_")]))
    return (embed_plot,)


@app.cell
def _(embed_plot):
    # Lasso/box selection, which sets the viewer's starting cluster further down.
    def _selected_pdb_ids():
        ids = []
        for pt in embed_plot.value or []:
            cd = pt.get("customdata")
            if isinstance(cd, (list, tuple)) and cd:
                ids.append(str(cd[0]))
        seen = set()
        return [i for i in ids if not (i in seen or seen.add(i))]

    umap_selection = _selected_pdb_ids()
    return (umap_selection,)


@app.cell
def _(active_info, cloud_path, load_pocket, np, qviz):
    # ---- The lattice the viewer's set-arithmetic modes rest on ---------------
    # Every pocket in a run is sampled on the same regular lattice, which is what
    # makes "difference from representative" and "consensus occupancy" exact rather
    # than interpolated. Never fatal: a run whose spacing cannot be established
    # still gets a viewer, just without those two modes.
    _clouds = [
        load_pocket(str(cloud_path(_pid)))[["x", "y", "z"]].to_numpy(dtype=float)
        for _pid in active_info["pdb_id"]
        if cloud_path(_pid).exists()
    ]
    grid_spacing = None
    if _clouds:
        try:
            grid_spacing = qviz.infer_spacing(np.vstack(_clouds))
        except ValueError:
            grid_spacing = None
    return (grid_spacing,)


@app.cell
def _(mo, qpages):
    qpages.on("Exploration", mo.md("""
    ## Pockets side by side
    Panels share framing and colour limits, so colour differences are real. Drag one
    to rotate all.
    """))
    return


@app.cell
def _(mo):
    panel_mode = mo.ui.dropdown(
        options=["one cluster", "every cluster's representative"],
        value="one cluster",
        label="Panels",
    )
    return (panel_mode,)


@app.cell
def _(FEATURE_COLS, active_info, mo, panel_mode, qcontrols, qpages, umap_selection):
    # ---- Viewer controls ----------------------------------------------------
    _clusters = [str(c) for c in sorted(active_info["cluster_id"].dropna().unique())]

    # A selection in the landscape wins over the default cluster.
    _default_cluster = _clusters[0] if _clusters else None
    for _s in umap_selection:
        _hit = active_info[active_info["pdb_id"] == _s]
        if len(_hit):
            _default_cluster = str(_hit.iloc[0]["cluster_id"])
            break

    cluster_sel = mo.ui.dropdown(options=_clusters, value=_default_cluster, label="Cluster")
    feature = mo.ui.dropdown(
        options=FEATURE_COLS,
        value=(FEATURE_COLS[0] if FEATURE_COLS else None),
        label="Feature",
    )
    n_cols = mo.ui.slider(start=1, stop=6, value=4, label="Columns")
    # Feature values are heavy-tailed (one run's `cp` peaks at 1679 against a 99th
    # percentile of 66), so clipping the ends is what keeps the ramp usable.
    clip_pct = mo.ui.slider(start=0, stop=10, step=0.5, value=2, label="Colour clip (%)")
    color_scheme = mo.ui.dropdown(
        options=["feature-aware", "perceptual (CVD-safe)"],
        value="feature-aware",
        label="Colour scheme",
    )
    # What the spheres encode. The last two are set operations on the shared
    # lattice, so they are exact rather than interpolated.
    colour_mode = mo.ui.dropdown(
        options=[
            "feature value",
            "difference from representative",
            "consensus occupancy",
        ],
        value="feature value",
        label="Spheres show",
    )
    sphere_radius = mo.ui.slider(start=0.1, stop=0.6, step=0.05, value=0.25, label="Pocket sphere radius (Å)")
    panel_px = mo.ui.slider(start=200, stop=520, step=20, value=320, label="Panel height (px)")
    protein_style = mo.ui.dropdown(
        options=["cartoon + pocket side chains", "cartoon only", "none"],
        value="cartoon + pocket side chains",
        label="Protein",
    )
    cartoon_opacity = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.65, label="Cartoon opacity"
    )
    context_radius = mo.ui.slider(
        start=6, stop=30, step=2, value=12, label="Protein context (Å)"
    )
    # Carbons take the class colour; 3Dmol applies the rest of the map by element,
    # so N stays blue, O red and S yellow in both classes.
    _swatches = {"orange": "#eb6834", "violet": "#7b5cd6", "cyan": "#22a7b8",
                 "magenta": "#d13f9a", "lime": "#7ab317"}
    observed_colour = mo.ui.dropdown(options=_swatches, value="orange",
                                     label="Contacted carbons")
    potential_colour = mo.ui.dropdown(options=_swatches, value="violet",
                                      label="Pocket-only carbons")
    show_all_sidechains = mo.ui.checkbox(
        value=True, label="Side chains for every interacting residue")
    show_ligand = mo.ui.checkbox(value=True, label="Ligand")
    # 3Dmol's *Carbon schemes recolour carbon only and leave CPK heteroatoms
    # alone, so N stays blue, O red and S yellow whichever hue you pick. Switch
    # off green when the feature ramp itself is green or teal.
    ligand_color = mo.ui.dropdown(
        options={
            "green": "greenCarbon",
            "magenta": "magentaCarbon",
            "cyan": "cyanCarbon",
            "yellow": "yellowCarbon",
            "orange": "orangeCarbon",
            "purple": "purpleCarbon",
        },
        value="green",
        label="Ligand carbons",
    )
    show_pocket = mo.ui.checkbox(value=True, label="Pocket cloud")
    # Every hoverable sphere gets its own intersection shape, so a big cluster
    # can feel sluggish — worth being able to turn off.
    hover_values = mo.ui.checkbox(value=True, label="Hover values")
    # Inlining the vendored 3Dmol build costs about half a megabyte per render
    # but needs no network, which matters on a locked-down machine.
    offline_viewer = mo.ui.checkbox(value=True, label="Offline viewer")

    # What you are looking at stays on screen; how it is drawn folds away.
    qpages.on("Exploration", qcontrols.panel(
        primary=[panel_mode]
        + ([cluster_sel] if panel_mode.value == "one cluster" else [])
        + [feature],
        groups={
            "Colour": [colour_mode, color_scheme, clip_pct],
            "Layout": [n_cols, panel_px, sphere_radius],
            "Protein": [protein_style, cartoon_opacity, context_radius,
                        show_all_sidechains, observed_colour, potential_colour],
            "Include": [show_ligand, ligand_color, show_pocket, hover_values,
                        offline_viewer],
        },
    ))
    return (
        cartoon_opacity,
        clip_pct,
        cluster_sel,
        color_scheme,
        colour_mode,
        context_radius,
        feature,
        hover_values,
        ligand_color,
        n_cols,
        observed_colour,
        offline_viewer,
        panel_px,
        potential_colour,
        protein_style,
        show_ligand,
        show_all_sidechains,
        show_pocket,
        sphere_radius,
    )


@app.cell
def _(active_info, cluster_sel, mo, panel_mode, qpages, qviz):
    # Which pockets are drawn: searchable, at most 40 at once; representatives first.
    _rows = qviz.grid.picker_rows(active_info, panel_mode.value, cluster_sel.value)
    _options = qviz.grid.cloud_options(_rows)
    _label_of = {v: k for k, v in _options.items()}
    _default = qviz.grid.default_cloud_selection(
        _rows, int(cluster_sel.value) if cluster_sel.value not in (None, "") else None, limit=40)
    if not _default:
        _default = list(_rows["pdb_id"].astype(str))[:40]
    pockets_shown = mo.ui.multiselect(options=_options, value=[_label_of[p] for p in _default if p in _label_of],
                                      label=f"Pockets shown (of {len(_rows)})", full_width=True)
    qpages.on("Exploration", pockets_shown)
    return (pockets_shown,)


@app.cell
def _(Path, active_info, cartoon_opacity, clip_pct, cloud_path, cluster_sel, color_scheme, colour_mode, context_radius, feature, grid_spacing, hover_values, json, ligand_color, load_pocket, mo, n_cols, np, observed_colour, offline_viewer, panel_mode, panel_px, pd, pockets_shown, potential_colour, protein_style, qpages, qviz, residue_picker, run_interactions, show_all_sidechains, show_ligand, show_pocket, sphere_radius):
    CONTACT_R = 4.0  # side chains are drawn for residues this close to the cloud

    # Before Featurise the clouds carry no feature column: drawn in one colour.
    feat = feature.value or "unfeaturised"
    across_clusters = panel_mode.value != "one cluster"
    _sizes = active_info["cluster_id"].value_counts().to_dict()

    if across_clusters:
        # One panel per cluster, showing the structure that stands for it: the
        # method's own exemplar where there is one, otherwise the member closest to
        # its cluster's exemplar, so no cluster is silently left out.
        _rows, _fallbacks = qviz.grid.representative_rows(active_info)
    else:
        mo.stop(cluster_sel.value is None, qpages.on("Exploration", mo.md("No cluster available.")))
        _fallbacks = []
        _rows = active_info[active_info["cluster_id"].astype(str) == str(cluster_sel.value)]

    # ---- Gather the pockets of this cluster ---------------------------------
    # Every member is read, so consensus and difference are over the whole cluster;
    # only the picked ones (at most 40) are drawn.
    all_pockets = []  # (pdb_id, metadata row, point cloud)
    missing = []
    for _, _row in _rows.iterrows():
        _fp = cloud_path(_row["pdb_id"])
        if _fp.exists():
            all_pockets.append((str(_row["pdb_id"]), _row, load_pocket(str(_fp))))
        else:
            missing.append(str(_row["pdb_id"]))

    mo.stop(not all_pockets, qpages.on("Exploration", mo.md(f"No pocket cloud found for cluster {cluster_sel.value}.")))
    _picked, _cap_note = qviz.grid.capped([str(v) for v in (pockets_shown.value or [])], 40)
    pockets = [p for p in all_pockets if p[0] in set(_picked)] or all_pockets[:40]
    if _cap_note:
        mo.output.append(mo.md(f"_{_cap_note}_"))

    # ---- What the spheres encode, and on what scale --------------------------
    # Each mode produces (coordinates, values) per panel. The difference and
    # consensus modes are set operations on the shared lattice, so they need no
    # interpolation; a site only one pocket occupies has no difference to show and
    # is dropped rather than filled in.
    mode = colour_mode.value
    if feature.value is None and mode == "difference from representative":
        mode = "feature value"
    # Within a cluster the reference is that cluster's own representative. Across
    # clusters it is the largest cluster's, so each panel reads as "how this
    # conformational family differs from the dominant one".
    if across_clusters:
        _rep_row = next(
            (r for _, r, _ in sorted(all_pockets, key=lambda t: -_sizes.get(t[1]["cluster_id"], 0))),
            None,
        )
    else:
        _rep_row = next((r for _, r, _ in all_pockets if bool(r.get("is_representative"))), None)
    if mode != "feature value" and grid_spacing is None:
        mode = "feature value"
        mo.output.append(
            mo.md("_No common lattice for these pockets, so showing feature values._")
        )
    if mode == "difference from representative" and _rep_row is None:
        mode = "feature value"
        mo.output.append(mo.md("_This cluster has no representative, so showing feature values._"))

    panel_values, value_label, ramp, domain, stops = qviz.grid.panel_values(
        all_pockets, mode, feat, _rep_row, grid_spacing, color_scheme.value, float(clip_pct.value))

    # ---- Shared framing box (pockets are pre-aligned) -----------------------
    _pts = np.vstack([p[["x", "y", "z"]].to_numpy(dtype=float) for _, _, p in pockets])
    bbox_pdb = qviz.pseudo_atom_pdb(qviz.bbox_corners(_pts), resname="BOX", chain="Z")

    # ---- Per-panel molecular models -----------------------------------------
    panels = qviz.grid.build_panels(
        pockets, panel_values, domain, across_clusters=across_clusters, sizes=_sizes,
        show_pocket=show_pocket.value, picks=residue_picker.value,
        run_interactions=run_interactions, protein_style=protein_style.value,
        show_all_sidechains=show_all_sidechains.value,
        context_radius=float(context_radius.value), contact_r=CONTACT_R,
        show_ligand=show_ligand.value)
    _html, grid_h = qviz.grid.grid_html(
        panels, bbox_pdb=bbox_pdb, n_cols=n_cols.value, panel_px=panel_px.value,
        sphere_radius=sphere_radius.value, stops=stops, value_label=value_label, ramp=ramp,
        domain=domain, hover=hover_values.value, cartoon_opacity=cartoon_opacity.value,
        ligand_color=ligand_color.value, observed_colour=observed_colour.value,
        potential_colour=potential_colour.value, offline=offline_viewer.value)

    _notes = qviz.grid.grid_notes(
        panels, across_clusters=across_clusters, fallbacks=_fallbacks, missing=missing,
        protein_style=protein_style.value, context_radius=context_radius.value,
        contact_r=CONTACT_R, run_interactions=run_interactions, mode=mode,
        value_label=value_label, picks=residue_picker.value)
    qpages.on("Exploration", mo.vstack(
        [
            mo.iframe(_html, height=f"{grid_h + 70}px"),
            mo.md("_" + " · ".join(_notes) + "_"),
        ]
    ))
    return


@app.cell
def _(mo, qpages):
    qpages.on("Exploration", mo.md("""
    ## Ligands
    One depiction per distinct molecule, most frequent first. Filter by cluster to
    link chemotypes to conformations.
    """))
    return


@app.cell
def _(actives_df, mo, pd, qpages):
    # ---- Canonical SMILES, computed once per run ----------------------------
    # RDKit parsing is the slow part and a screening table can run to a few
    # thousand rows, so it happens here rather than in the cell the cluster
    # selector re-runs.
    mo.stop(
        actives_df is None or actives_df.empty,
        qpages.on("Exploration", mo.md("No `screening/actives.smi` for this site, so there is no ligand chemistry to draw.")),
    )

    from rdkit import Chem as _Chem, RDLogger as _RDLogger

    _RDLogger.DisableLog("rdApp.*")  # unparsed rows are counted below, not printed

    _smiles_col = "SMILES" if "SMILES" in actives_df.columns else actives_df.columns[0]
    _pdb_col = "PDB_ID" if "PDB_ID" in actives_df.columns else actives_df.columns[1]
    _code_col = "Ligand_Name" if "Ligand_Name" in actives_df.columns else actives_df.columns[2]

    _canonical_of = {}
    for _smiles in actives_df[_smiles_col].dropna().astype(str).unique():
        _mol = _Chem.MolFromSmiles(_smiles)
        _canonical_of[_smiles] = None if _mol is None else _Chem.MolToSmiles(_mol)

    ligand_rows = pd.DataFrame(
        {
            "pdb_id": actives_df[_pdb_col].astype(str).str.upper(),
            "code": actives_df[_code_col].astype(str).str.upper(),
            "canonical": actives_df[_smiles_col].astype(str).map(_canonical_of),
        }
    )
    n_unparsed = int(ligand_rows["canonical"].isna().sum())
    ligand_rows = ligand_rows.dropna(subset=["canonical"])
    return ligand_rows, n_unparsed


@app.cell
def _(active_info, mo, qcontrols, qpages):
    _clusters = [str(c) for c in sorted(active_info["cluster_id"].dropna().unique())]
    ligand_cluster = mo.ui.dropdown(
        options=["all clusters", *_clusters], value="all clusters", label="Cluster"
    )
    ligand_cols = mo.ui.slider(start=1, stop=6, value=4, label="Columns")
    # A screening table can name hundreds of compounds and every panel is drawn as
    # SVG paths, so the cap is what keeps a large run from rendering megabytes.
    max_ligands = mo.ui.slider(start=4, stop=120, step=4, value=24, label="Max molecules")
    show_ligand_props = mo.ui.checkbox(value=True, label="Properties table")
    qpages.on("Exploration", qcontrols.panel(
        primary=[ligand_cluster],
        groups={"Layout": [ligand_cols, max_ligands],
                "Include": [show_ligand_props]},
    ))
    return ligand_cluster, ligand_cols, max_ligands, show_ligand_props


@app.cell
def _(active_info, ligand_cluster, ligand_cols, ligand_rows, max_ligands, mo, n_unparsed, qpages, qviz, show_ligand_props):
    # The table can reach beyond the structures of this run, and a cluster is a
    # subset of those, so both are filtered on the active grouping.
    _rows, _outside, _cluster_of = qviz.ligands.in_scope(ligand_rows, active_info, ligand_cluster.value)
    mo.stop(
        _rows.empty,
        qpages.on("Exploration", mo.md(f"_No structure in cluster {ligand_cluster.value} was solved with a ligand._")),
    )
    _unique = qviz.ligands.distinct_molecules(_rows, _cluster_of)
    _shown = _unique[: int(max_ligands.value)]
    _svg, _mols = qviz.ligands.depictions(_shown, n_cols=int(ligand_cols.value),
                                          all_clusters=ligand_cluster.value == "all clusters")
    _notes = qviz.ligands.notes(_shown, _unique, _rows, _outside, n_unparsed)
    qpages.on("Exploration", mo.vstack(
        [
            mo.Html(f'<div style="overflow-x:auto">{_svg}</div>'),
            mo.md("_" + " · ".join(_notes) + "_"),
        ]
        + ([mo.ui.table(qviz.ligands.properties(_shown, _mols), selection=None, pagination=True)]
           if show_ligand_props.value else [])
    ))
    return

@app.cell
def _(mo, qpages):
    qpages.on("Exploration", mo.md("""
    ## Cluster distances
    The median distance between every pair of clusters, as a site percentile (dark = close).
    Pick two clusters to see their members side by side.
    """))
    return


@app.cell
def _(active_info, cluster_matrices, mo, qpages, qviz):
    _kinds = ["shape"] + (["chemistry", "combined"] if cluster_matrices is not None
                          and cluster_matrices.chem is not None else [])
    dist_kind_pick = mo.ui.dropdown(options=_kinds, value="shape", label="Distance")
    cluster_names = qviz.clusters.cluster_labels_of(active_info)
    _options = {name: str(c) for c, name in sorted(cluster_names.items()) if c >= 0}
    _first = list(_options)
    pair_a = mo.ui.dropdown(options=_options, value=_first[0] if _first else None, label="Cluster")
    pair_b = mo.ui.dropdown(options=_options, value=_first[min(1, len(_first) - 1)] if _first else None,
                            label="against")
    qpages.on("Exploration", dist_kind_pick)
    return cluster_names, dist_kind_pick, pair_a, pair_b


@app.cell
def _(active_info, cluster_matrices, cluster_names, cluster_settings, dist_kind_pick, mo, pd, qpages, qviz):
    # The k × k map follows the distance kind only, so picking a pair below does not recompute it.
    mo.stop(cluster_matrices is None, qpages.on("Exploration", mo.md("No distance matrix.")))
    _full = {"shape": cluster_matrices.shape, "chemistry": cluster_matrices.chem,
             "combined": cluster_matrices.combined}[dist_kind_pick.value]
    site_M = pd.DataFrame(_full, index=cluster_matrices.ids, columns=cluster_matrices.ids)
    site_labels = active_info.assign(pdb_id=active_info["pdb_id"].astype(str)).set_index("pdb_id")["cluster_id"]
    site_labels = site_labels.reindex(site_M.index).dropna().astype(int)
    site_M = site_M.loc[site_labels.index, site_labels.index]
    _shape, _chem = qviz.clusters.display_ids(active_info)
    _by_pair = qviz.style.cluster_colours(_shape, _chem, int(cluster_settings.min_cluster_size))
    _colours = {int(c): _by_pair.get((int(sh), int(ch)), qviz.style.GREY)
                for c, sh, ch in zip(active_info["cluster_id"], _shape, _chem) if c == c}
    _cmap = qviz.matrix.cluster_map(site_M, site_labels)
    qpages.on("Exploration", qviz.matrix.cluster_map_figure(_cmap, _colours, names={c: cluster_names.get(int(c), str(c))
                                                          for c in _cmap.index}))
    return site_M, site_labels


@app.cell
def _(mo, pair_a, pair_b, qpages, qviz, site_M, site_labels):
    _view = mo.md("")
    if pair_a.value is not None and pair_b.value is not None:
        _block = qviz.matrix.member_block(site_M, site_labels, int(pair_a.value), int(pair_b.value))
        _view = qviz.style.apply(qviz.matrix.heatmap(_block, list(_block.index), [], "as loaded"))
    qpages.on("Exploration", mo.vstack([mo.hstack([pair_a, pair_b], justify="start", gap=2), _view]))
    return

@app.cell
def _(mo, qpages):
    qpages.on("Exploration", mo.md("""
    ## Interaction fingerprint
    What the ligands touch, and what merely lines the pocket. Percentages on the
    left are over the ligand-bound structures; on the right, the pale bar is over
    every structure, since an apo pocket still has a lining. Rows are the same in
    both panels, and in every cluster.
    """))
    return


@app.cell
def _(active_info, mo, qcontrols, qpages, run_interactions):
    mo.stop(run_interactions is None,
            qpages.on("Exploration", mo.md("No `observed_interactions.h5` or `potential_interactions.h5` in this run.")))

    _clusters = [str(c) for c in sorted(active_info["cluster_id"].dropna().unique())]
    fingerprint_scope = mo.ui.dropdown(
        options=["all clusters", *_clusters], value="all clusters", label="**Show**")
    potential_mode = mo.ui.dropdown(
        options=["every pocket residue", "never contacted"],
        value="every pocket residue", label="Right panel")
    min_frequency = mo.ui.slider(start=0, stop=100, step=5, value=0,
                                 label="Hide below (%)", show_value=True)
    row_px = mo.ui.slider(start=12, stop=40, step=2, value=24, label="Row height (px)")

    qpages.on("Exploration", qcontrols.panel(
        primary=[fingerprint_scope],
        groups={"Rows": [potential_mode, min_frequency, row_px]},
    ))
    return fingerprint_scope, min_frequency, potential_mode, row_px


@app.cell
def _(active_info, fingerprint_scope, min_frequency, mo, potential_mode, qpages, qviz, row_px, run_interactions):
    mo.stop(run_interactions is None, qpages.on("Exploration", mo.md("")))

    if fingerprint_scope.value == "all clusters":
        _members = list(active_info["pdb_id"].astype(str))
        _label = "all clusters"
    else:
        _members = list(
            active_info[active_info["cluster_id"].astype(str) == str(fingerprint_scope.value)]
            ["pdb_id"].astype(str)
        )
        _label = f"cluster {fingerprint_scope.value}"

    _freqs = qviz.interactions.filter_residues(
        qviz.interactions.cluster_frequencies(run_interactions, _members, label=_label),
        min_percent=float(min_frequency.value),
        only_unused=potential_mode.value == "never contacted",
    )

    mo.stop(not _freqs.residues,
            qpages.on("Exploration", mo.md(f"_Nothing left to plot for {_label} at this cutoff._")))

    fingerprint_plot = mo.ui.plotly(
        qviz.interactions.observed_figure(_freqs, height_per_row=int(row_px.value)))
    _right = mo.ui.plotly(
        qviz.interactions.potential_figure(_freqs, height_per_row=int(row_px.value)))

    _notes = [f"{len(_freqs.residues)} residues",
              f"{_freqs.n_holo} of {_freqs.n_members} structures carry a ligand"]
    if run_interactions.folded:
        _notes.append(f"folded into \u201cOther\u201d: {', '.join(run_interactions.folded)}")

    qpages.on("Exploration", mo.vstack([
        mo.hstack([fingerprint_plot, _right], widths=[1, 1], gap=0.5),
        mo.md("_" + " \u00b7 ".join(_notes) + "_"),
    ]))
    return (fingerprint_plot,)


@app.cell
def _(fingerprint_plot, mo, qpages, run_interactions):
    # Residues picked out here are drawn as highlighted sticks in every panel of
    # the viewer above, coloured by whether they are contacted or only line the
    # pocket. Box-select the bars, or pick from the list.
    _selected = []
    for _pt in fingerprint_plot.value or []:
        _cd = _pt.get("customdata")
        if isinstance(_cd, (list, tuple)) and _cd:
            _selected.append(str(_cd[0]))
        elif _pt.get("y"):
            _selected.append(str(_pt["y"]))

    residue_picker = mo.ui.multiselect(
        options=list(run_interactions.residues) if run_interactions else [],
        value=sorted(dict.fromkeys(_selected)),
        label="Highlight residues",
    )
    qpages.on("Exploration", mo.hstack([residue_picker], justify="start"))
    return (residue_picker,)


@app.cell
def _(active_info, mo, qexplore, qpages, qviz, run_interactions):
    # Contacted share over each cluster's ligand-bound members; hover gives the lining share too.
    _names = qviz.clusters.cluster_labels_of(active_info) if "cluster_id" in active_info else {}
    _members = qexplore.cluster_members(active_info, _names) if _names else {}
    qpages.on("Exploration", mo.vstack([
        mo.md("## Residue contacts by cluster\nShare of each cluster's ligand-bound structures whose ligand "
              "contacts the residue."),
        qexplore.residue_heatmap(qexplore.residue_profile(run_interactions, _members)),
    ]))
    return


@app.cell
def _(mo):
    # Bumped by Finalise, so the status and the Screening guard re-read final/; Finalise's outcome.
    get_final_version, set_final_version = mo.state(0)
    get_final_message, set_final_message = mo.state(None)
    return get_final_message, get_final_version, set_final_message, set_final_version


@app.cell
def _(active_info, get_final_message, get_final_version, mo, qfinal, qpages, run_dir, site_dir):
    _ = (active_info, get_final_version())   # re-read after Apply, a representative swap, or Finalise
    _status = qfinal.final_status(run_dir, site_dir.name)
    _history = site_dir / "final" / "history"
    _n_history = len([p for p in _history.iterdir() if p.is_dir()]) if _history.is_dir() else 0
    if _status == "current":
        import json as _json

        _when = _json.loads((site_dir / "final" / "settings.json").read_text(encoding="utf-8"))["finalised_at"]
        _line = f"Finalised {_when} and current · {_n_history} earlier version(s) in `final/history/`."
    elif _status == "stale":
        _line = "Finalised, but the clustering or a representative changed since: finalise again."
    else:
        _line = "Not finalised yet."
    finalise_button = mo.ui.run_button(label="Finalise site")
    qpages.on("Exploration", mo.vstack([
        mo.md("## Finalise\nFreezes this site's clustering, representatives, residue profile and actives into "
              f"`{site_dir / 'final'}`; Screening reads it.\n\n**{_line}**"),
        finalise_button,
        mo.md(qpages.shown(get_final_message(), qpages.clustering_stamp(site_dir))),
    ]))
    return (finalise_button,)


@app.cell
def _(finalise_button, mo, qfinal, qpages, qsites, run_dir, set_final_message, set_final_version, site_dir):
    mo.stop(not finalise_button.value)
    try:
        _folder = qfinal.finalise_site(run_dir, site_dir.name)
        _text = f"_Finalised into `{_folder}`._"
    except qsites.SiteError as _e:
        _text = f"_{_e}_"
    # Hidden again once the clustering changes (the final is stale then) or on another site.
    set_final_message(qpages.stamped(_text, qpages.clustering_stamp(site_dir)))
    set_final_version(lambda v: v + 1)
    return


@app.cell
def _(active_info, get_final_version, mo, qfinal, qpages, run_dir, site_dir):
    _ = (active_info, get_final_version())
    _status = qfinal.final_status(run_dir, site_dir.name)
    _why = {"missing": " It has no final metadata yet.",
            "stale": " Its clustering or a representative changed since it was finalised."}.get(_status, "")
    qpages.on("Screening", mo.callout(mo.md(f"Finalise this site on the Exploration page.{_why}"), kind="warn")
              if _status != "current" else mo.md(f"_Screening {site_dir.name} from its final metadata._"))
    return


@app.cell
def _(mo):
    # Outcomes of the Screening buttons, by action, each kept with its site; a counter that re-reads the task
    # folder after one; a counter that redraws the poses once new ones may be there.
    get_screen_messages, set_screen_messages = mo.state({})
    get_screen_version, set_screen_version = mo.state(0)
    get_pose_version, set_pose_version = mo.state(0)
    return (get_pose_version, get_screen_messages, get_screen_version, set_pose_version, set_screen_messages,
            set_screen_version)


@app.cell
def _(active_info, get_final_version, mo, qfinal, qpages, qsubmit, run_dir, site_dir):
    _ = (active_info, get_final_version())
    # Everything below works from the site's final metadata; the guard above says what to do first.
    mo.stop(qfinal.final_status(run_dir, site_dir.name) != "current")
    from qpocket.screen_config import load_screen_config

    # The site's screen_config.yaml, else the run's, else the defaults: the page, Prepare and Analyse share it.
    screen_cfg_path = qsubmit.screen_config_path(run_dir, site_dir.name)
    screen_cfg = load_screen_config(str(screen_cfg_path) if screen_cfg_path else None)
    screen_final = qfinal.read_final(run_dir, site_dir.name)
    crossdock_folder = site_dir / "screening" / "crossdock"
    run_type = mo.ui.radio(options=["Cross-docking"], value="Cross-docking", label="**Run type**")
    qpages.on("Screening", mo.vstack([
        mo.md("## Screening"),
        run_type,
        mo.md("<span style='color:#9aa3ad'>Active/decoy benchmark — not implemented yet</span>"),
        mo.md(f"_Settings from `{screen_cfg_path}`._" if screen_cfg_path else
              f"_Default settings; put a `screen_config.yaml` in `{site_dir / 'screening'}` or `{run_dir}` "
              "to change them (engines, analysis, `remote:` for a VM)._"),
    ]))
    return crossdock_folder, screen_cfg, screen_cfg_path, screen_final


@app.cell
def _(mo, qscreen, screen_cfg, screen_final):
    _ligs = screen_final["ligands"]
    _labels = {f"{r.ligand_key} · {r.resname} · {r.cluster}": r.ligand_key for r in _ligs.itertuples()}
    _default = set(qscreen.default_ligands(_ligs, screen_final["clusters"], limit=40))
    box_mode = mo.ui.radio(options={"Interacting residues": "interacting_residues",
                                    "Representatives' clouds": "representative_pockets"},
                           value=("Interacting residues" if screen_cfg.crossdock.box_mode == "interacting_residues"
                                  else "Representatives' clouds"), label="**Box**")
    box_padding = mo.ui.slider(start=0.0, stop=15.0, step=0.5, value=float(screen_cfg.crossdock.box_padding),
                               label="Padding (Å)", show_value=True, debounce=True)
    box_ligands = mo.ui.multiselect(options=_labels, value=[k for k, v in _labels.items() if v in _default],
                                    label="Ligands shown (at most 40)", max_selections=40)
    return box_ligands, box_mode, box_padding


@app.cell
def _(box_ligands, box_mode, box_padding, mo, qbox, qpages, qscreen, run_dir, site_dir):
    # The viewer only: button outcomes are shown by the cell below, so they never redraw it.
    box_preview, _error = None, None
    try:
        _make = qbox.interacting_box if box_mode.value == "interacting_residues" else qbox.cloud_box
        box_preview = _make(run_dir, site_dir.name, float(box_padding.value))
    except qbox.BoxError as _e:
        _error = str(_e)
    if box_preview is None:
        _view = mo.callout(mo.md(_error), kind="warn")
    elif box_preview.mode == "interacting_residues":
        _files = qscreen.ligand_files(run_dir, list(box_ligands.value))
        _view = mo.iframe(qscreen.ligand_box_scene(box_preview, _files), height="600px")
    else:
        _view = mo.iframe(qscreen.cloud_box_scene(box_preview), height="600px")
    qpages.on("Screening", mo.vstack([
        mo.md("### Box"),
        mo.hstack([box_mode, box_padding], justify="start", align="end"),
        box_ligands if box_mode.value == "interacting_residues" else mo.md(""),
        _view,
    ]))
    return (box_preview,)


@app.cell
def _(box_preview, get_screen_messages, get_screen_version, mo, qbox, qpages, run_dir, site_dir):
    _ = get_screen_version()
    _saved, _status = qbox.saved_box(run_dir, site_dir.name)
    _line = {"none": "No box saved yet; preparation uses the configured box.",
             "invalid": "The saved box.json is unreadable or incomplete; save the box again.",
             "stale": "The saved box belongs to an earlier final (or older contacts); save it again.",
             "current": f"Saved box: {(_saved or {}).get('mode')}, padding {(_saved or {}).get('padding')} Å, "
                        f"{(_saved or {}).get('saved_at')}."}[_status]
    save_box_button = mo.ui.run_button(label="Save box", disabled=box_preview is None)
    qpages.on("Screening", mo.vstack([
        mo.hstack([save_box_button, mo.md(f"_{_line}_")], justify="start", align="center"),
        mo.md(qpages.shown(get_screen_messages().get("box"), str(site_dir))),
    ]))
    return (save_box_button,)


@app.cell
def _(box_preview, get_screen_messages, mo, qbox, qpages, run_dir, save_box_button, set_screen_messages,
      set_screen_version, site_dir):
    mo.stop(not save_box_button.value or box_preview is None)
    _path = qbox.save_box(run_dir, site_dir.name, box_preview)
    set_screen_messages({**get_screen_messages(), "box": qpages.stamped(f"_Saved `{_path}`._", str(site_dir))})
    set_screen_version(lambda v: v + 1)
    return


@app.cell
def _(mo, qsubmit, run_dir, screen_cfg, site_dir):
    _found = qsubmit.engines(screen_cfg)
    _labels = {(name if path else f"{name} (not installed)"): name for name, path in _found.items()}
    _default = next((k for k, v in _labels.items() if v == screen_cfg.docking.engine and _found[v]),
                    next((k for k, v in _labels.items() if _found[v]), None))
    dock_engine = mo.ui.dropdown(options=_labels, value=_default, label="Engine")
    dock_exhaustiveness = mo.ui.number(start=1, stop=128, step=1, value=int(screen_cfg.docking.exhaustiveness),
                                       label="Exhaustiveness")
    dock_modes = mo.ui.number(start=1, stop=50, step=1, value=int(screen_cfg.docking.num_modes), label="Modes")
    dock_cpus = mo.ui.number(start=1, stop=128, step=1, value=int(screen_cfg.docking.cpus), label="CPUs per docking")
    prepare_button = mo.ui.run_button(label="Prepare")
    dock_installed = _found
    try:
        dock_pairs = qsubmit.pair_count(run_dir, site_dir.name)
    except FileNotFoundError:
        dock_pairs = (0, 0, 0)
    return dock_cpus, dock_engine, dock_exhaustiveness, dock_installed, dock_modes, dock_pairs, prepare_button


@app.cell
def _(dock_cpus, dock_engine, dock_exhaustiveness, dock_modes, dock_pairs, get_screen_messages, mo,
      prepare_button, qpages, site_dir):
    _r, _c, _p = dock_pairs
    qpages.on("Screening", mo.vstack([
        mo.md("### Docking"),
        mo.hstack([dock_engine, dock_exhaustiveness, dock_modes, dock_cpus], justify="start", align="end"),
        mo.md(f"**{_r} receptors × {_c} compounds = {_p} dockings** (every distinct compound into every "
              "representative, redocking included)."),
        prepare_button,
        mo.md(qpages.shown(get_screen_messages().get("prepare"), str(site_dir))),
    ]))
    return


@app.cell
def _(dock_cpus, dock_engine, dock_exhaustiveness, dock_installed, dock_modes, get_screen_messages, mo,
      prepare_button, qcrossdock, qpages, screen_cfg_path, set_screen_messages, set_screen_version, site_dir):
    mo.stop(not prepare_button.value)

    if dock_engine.value is None or not dock_installed.get(dock_engine.value):
        _msg = f"_{dock_engine.value or 'No engine'} is not installed here; pick an installed one._"
    else:
        with mo.status.spinner(title="Preparing receptors, ligands and the task list…"):
            try:
                _plan = qcrossdock.prepare(site_dir, str(screen_cfg_path) if screen_cfg_path else None, docking={
                    "engine": dock_engine.value, "exhaustiveness": int(dock_exhaustiveness.value),
                    "num_modes": int(dock_modes.value), "cpus": int(dock_cpus.value)})
                _warn = "".join(f"\n- {w}" for w in _plan.warnings)
                _msg = f"_Prepared `{_plan.jobs_csv}`._" + (f"\n\nWarnings:{_warn}" if _warn else "")
            except Exception as _e:
                _msg = f"_Preparation failed: {type(_e).__name__}: {_e}_"
    set_screen_messages({**get_screen_messages(), "prepare": qpages.stamped(_msg, str(site_dir))})
    set_screen_version(lambda v: v + 1)
    return


@app.cell
def _(crossdock_folder, get_screen_version, mo):
    _ = get_screen_version()
    task_ready = (crossdock_folder / "docking_jobs.csv").exists()
    run_workers = mo.ui.number(start=0, stop=512, step=1, value=0, label="Parallel dockings (0 = half the cores)")
    run_start = mo.ui.run_button(label="Start", disabled=not task_ready)
    run_refresh = mo.ui.run_button(label="Refresh progress", disabled=not task_ready)
    run_stop = mo.ui.run_button(label="Stop", kind="danger", disabled=not task_ready)
    vm_push = mo.ui.run_button(label="Push & start", disabled=not task_ready)
    vm_poll = mo.ui.run_button(label="Poll", disabled=not task_ready)
    vm_pull = mo.ui.run_button(label="Pull poses", disabled=not task_ready)
    return run_refresh, run_start, run_stop, run_workers, task_ready, vm_poll, vm_pull, vm_push


@app.cell
def _(crossdock_folder, dock_cpus, get_screen_messages, mo, qpages, qsubmit, run_refresh, run_start, run_stop,
      run_workers, screen_cfg, site_dir, task_ready, vm_poll, vm_pull, vm_push):
    _ = (run_refresh.value, get_screen_messages())  # Start and Stop leave a message: show the new state too
    if not task_ready:
        _here = mo.md("_Prepare first._")
    else:
        _prog = qsubmit.local_progress(crossdock_folder)
        _state = "running" if _prog["running"] else ("started " + str(_prog["started_at"]) if _prog["started_at"]
                                                       else "not started here")
        _here = mo.vstack([mo.hstack([run_workers, run_start, run_refresh, run_stop], justify="start", align="end"),
                           mo.md(f"**{_prog['done']} / {_prog['total']}** dockings done · {_state}")])
    _sbatch = crossdock_folder / "run_docking.sbatch"
    _slurm = (mo.vstack([mo.md("Scripts are written; copy the folder to the cluster and submit:"),
                         mo.plain_text(f"cd {crossdock_folder} && sbatch run_docking.sbatch")])
              if _sbatch.exists() else mo.md("_Prepare first; it writes the SLURM script._"))
    _remote = screen_cfg.remote
    if not _remote.host:
        _vm = mo.md("_No `remote:` block in screen_config.yaml (host, directory, setup). Q-Pocket never starts "
                    "or stops a VM._")
    else:
        _vm = mo.vstack([
            mo.md(f"Host `{_remote.host}`, folder `{_remote.directory}/{site_dir.name}` · Poll also asks Azure "
                  "for the VM's power state when `azure_vm` is set."),
            mo.hstack([vm_push, vm_poll, vm_pull], justify="start"),
        ])
    qpages.on("Screening", mo.vstack([
        mo.md("### Run"),
        mo.accordion({"Run here": _here, "SLURM (writes scripts only)": _slurm, "VM over SSH": _vm}),
        mo.md(qpages.shown(get_screen_messages().get("run"), str(site_dir))),
    ]))
    return


@app.cell
def _(crossdock_folder, dock_cpus, get_screen_messages, mo, qpages, qsubmit, run_start, run_stop, run_workers,
      screen_cfg, set_pose_version, set_screen_messages, site_dir, vm_poll, vm_pull, vm_push):
    mo.stop(not (run_start.value or run_stop.value or vm_push.value or vm_poll.value or vm_pull.value))
    import subprocess as _subprocess

    try:
        if run_start.value:
            _rec = qsubmit.start_local(crossdock_folder, int(run_workers.value), int(dock_cpus.value))
            _msg = f"_Started here (pid {_rec['pid']}, {_rec['n_jobs']} dockings); Refresh progress to follow it._"
        elif run_stop.value:
            _msg = ("_Stopped the run started here._" if qsubmit.stop_local(crossdock_folder)
                    else "_No run started here is running._")
        else:
            _cmds = qsubmit.remote_commands(crossdock_folder, site_dir.name, screen_cfg.remote,
                                            int(run_workers.value), int(dock_cpus.value))
            _keys = ["push", "start"] if vm_push.value else ["poll"] if vm_poll.value else ["pull"]
            _outs = []
            for _k in _keys:
                _r = _subprocess.run(_cmds[_k], capture_output=True, text=True, timeout=600)
                if _r.returncode != 0:
                    raise RuntimeError(f"{_k} failed: {_r.stderr.strip()[:400]}")
                _outs.append(_r.stdout)
            if vm_poll.value:
                _st = qsubmit.remote_status(_outs[0], site_dir.name)
                _power = qsubmit.vm_power_state(screen_cfg.remote)
                _others = f" · **other jobs on the VM:** {', '.join(_st['other_jobs'])}" if _st["other_jobs"] else ""
                _msg = (f"_{_st['poses']} pose files there · our session "
                        f"{'running' if _st['own_session'] else 'not running'}"
                        + (f" · VM {_power}" if _power else "") + f"_{_others}")
            else:
                _msg = "_Pushed and started._" if vm_push.value else "_Pulled the poses._"
                if vm_pull.value:
                    set_pose_version(lambda v: v + 1)
    except Exception as _e:
        _msg = f"_{type(_e).__name__}: {_e}_"
    set_screen_messages({**get_screen_messages(), "run": qpages.stamped(_msg, str(site_dir))})
    return


@app.cell
def _(mo, screen_cfg):
    analyse_threshold = mo.ui.number(start=0.1, stop=10.0, step=0.1, value=float(screen_cfg.analysis.rmsd_threshold),
                                     label="Success threshold (Å)")
    analyse_button = mo.ui.run_button(label="Analyse")
    get_screen_result, set_screen_result = mo.state(None)
    return analyse_button, analyse_threshold, get_screen_result, set_screen_result


@app.cell
def _(analyse_button, analyse_threshold, get_screen_messages, mo, qcrossdock, qpages, screen_cfg_path,
      set_screen_messages, set_screen_result, site_dir):
    mo.stop(not analyse_button.value)

    try:
        with mo.status.spinner(title="Scoring poses against the crystal ligands…"):
            set_screen_result(qcrossdock.analyze(site_dir, str(screen_cfg_path) if screen_cfg_path else None,
                                                 analysis={"rmsd_threshold": float(analyse_threshold.value)}))
        _msg = ""
    except FileNotFoundError as _e:
        _msg = f"_Nothing to analyse yet: {_e}_"
    except Exception as _e:  # a broken pose file or table must not leave the button silent
        set_screen_result(None)
        _msg = f"_Analysis failed: {type(_e).__name__}: {_e}_"
    set_screen_messages({**get_screen_messages(), "analyse": qpages.stamped(_msg, str(site_dir))})
    return


@app.cell
def _(analyse_button, analyse_threshold, get_screen_messages, get_screen_result, mo, qpages, qscreen, site_dir):
    _res = get_screen_result()
    if _res is None or not len(_res.pairs):
        _view = mo.md("_No analysed poses yet._")
    else:
        _view = mo.vstack([
            mo.md("Top-1 and top-k success per receptor, per ligand and overall:"),
            mo.ui.table(_res.success, selection=None, pagination=True),
            qscreen.matrix_figure(qscreen.success_matrix(_res.pairs), float(analyse_threshold.value)),
        ])
    qpages.on("Screening", mo.vstack([
        mo.md("### Analysis"),
        mo.hstack([analyse_threshold, analyse_button], justify="start", align="end"),
        mo.md(qpages.shown(get_screen_messages().get("analyse"), str(site_dir))),
        _view,
    ]))
    return


@app.cell
def _(crossdock_folder, get_screen_version, mo, pd, task_ready):
    _ = get_screen_version()
    _jobs = pd.read_csv(crossdock_folder / "docking_jobs.csv") if task_ready else pd.DataFrame(
        columns=["representative", "active_id"])
    pose_receptor = mo.ui.dropdown(options=sorted(_jobs["representative"].astype(str).unique()), label="Receptor")
    pose_ligand = mo.ui.dropdown(options=sorted(_jobs["active_id"].astype(str).unique()), label="Ligand")
    pose_top = mo.ui.number(start=1, stop=20, step=1, value=3, label="Top poses")
    return pose_ligand, pose_receptor, pose_top


@app.cell
def _(analyse_threshold, crossdock_folder, get_pose_version, mo, pd, pose_ligand, pose_receptor, pose_top, qpages,
      qscreen, run_refresh, screen_final, task_ready):
    _ = (get_pose_version(), run_refresh.value)  # Pull and Refresh progress may bring new poses
    if not task_ready:
        _view = mo.md("_Prepare and run first._")
    elif pose_receptor.value is None or pose_ligand.value is None:
        _view = mo.md("_Pick a receptor and a ligand._")
    else:
        _jobs = pd.read_csv(crossdock_folder / "docking_jobs.csv", dtype={"engine": str})
        _match = _jobs[(_jobs["representative"].astype(str) == pose_receptor.value)
                       & (_jobs["active_id"].astype(str) == pose_ligand.value)]
        if _match.empty:
            _view = mo.md("_That pair was not docked._")
        else:
            _profile = screen_final["residue_profile"]
            _residues = list(_profile.loc[pd.to_numeric(_profile["contacted_pct"], errors="coerce") > 0, "residue"]
                             .drop_duplicates()) if len(_profile) else []
            _html, _table = qscreen.pose_scene(crossdock_folder, _match.iloc[0], int(pose_top.value),
                                               float(analyse_threshold.value), _residues)
            _view = mo.vstack([mo.iframe(_html, height="560px"),
                               mo.ui.table(_table, selection=None) if len(_table) else mo.md("")])
    qpages.on("Screening", mo.vstack([
        mo.md("### Poses"),
        mo.hstack([pose_receptor, pose_ligand, pose_top], justify="start", align="end"),
        _view,
    ]))
    return


if __name__ == "__main__":
    app.run()
