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

    return Path, json, lru_cache, mo, np, pd, px, qcontrols, qsites, qviz


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
def _(Path, mo, run_selector):
    mo.stop(run_selector.value is None, mo.md("No run selected / none found."))
    run_dir = Path(run_selector.value)
    return (run_dir,)


@app.cell
def _(mo, pd, qsites, regrouped, run_dir):
    # regrouped is None until Regroup sites runs; depending on it re-reads the table.
    _ = regrouped
    try:
        sites_df = qsites.site_table(run_dir)
    except qsites.SiteError:
        sites_df = pd.DataFrame(columns=["site_id", "n_holo_structures", "n_ligands", "extent", "resnames"])
    _labels = {f"{r.site_id} · {r.n_holo_structures} holo · {r.resnames}": r.site_id
               for r in sites_df.itertuples()}
    site_picker = mo.ui.dropdown(options=_labels, value=next(iter(_labels), None),
                                 label="**Binding site**")
    mo.vstack([
        mo.md("## Binding sites\nEvery ligand-occupied site of the run. Everything below reads "
              "the site picked here."),
        site_picker,
        mo.ui.table(sites_df[["site_id", "n_holo_structures", "n_ligands", "extent", "resnames"]],
                    selection=None, pagination=True),
    ])
    return site_picker, sites_df


@app.cell
def _(mo):
    # The last site step's outcome; setting it re-reads the files that step rewrote.
    get_site_steps, set_site_steps = mo.state(None)
    return get_site_steps, set_site_steps


@app.cell
def _(Path, get_site_steps, mo, pd, run_dir, site_picker):
    mo.stop(site_picker.value is None, mo.md("No ligand-bound site was found in this run."))

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

    mo.md(
        f"""
        `{site_dir}` — structures: **{len(info)}** · clusters:
        **{info['cluster_id'].nunique()}** · representatives:
        **{int(info['is_representative'].sum())}** · features:
        `{', '.join(FEATURE_COLS) or 'none found'}`
        """
    )
    return FEATURE_COLS, actives_df, cloud_path, dist_df, dist_kind, info, site_dir


@app.cell
def _(info, mo, qviz, run_dir, site_dir, sites_df):
    # The site's ligands over the frame protein, every site's centre as a sphere
    # (this one dark, the others faint), through the pocket grid's own page.
    _centres = sites_df[["centre_x", "centre_y", "centre_z"]].to_numpy(dtype=float)
    _here = (sites_df["site_id"] == site_dir.name).to_numpy(dtype=float)
    _ligands = site_dir / "ligands.pdb"
    # shortcut: the first aligned structure stands in for the frame; every one is superposed onto it.
    _frame = next((run_dir / f for f in info.get("aligned_file", [])
                   if isinstance(f, str) and f and (run_dir / f).exists()), None)
    _ramp = qviz.RampSpec("sequential", "Blues", ("other sites", "this site"))
    _domain = qviz.ColourDomain(0.15, 1.0, None)
    _stops = qviz.ramp_stops(_ramp.scale)
    _panel = {
        "id": site_dir.name, "title": f"{site_dir.name} · ligands of every holo structure",
        "cloud": qviz.pseudo_atom_pdb(_centres, b=qviz.scale_to_percent(_here, _domain)),
        "cartoon": _frame.read_text(encoding="utf-8") if _frame else "",
        "side": "", "highlights": {}, "n_forced": 0,
        "ligand": qviz.ligand_pdb(str(_ligands)) if _ligands.exists() else "",
        "vals": [round(float(v), 3) for v in _here], "n_points": len(_centres), "n_ctx": 0,
    }
    _html, _h = qviz.grid.grid_html(
        [_panel], bbox_pdb=qviz.pseudo_atom_pdb(qviz.bbox_corners(_centres, pad=10.0), resname="BOX", chain="Z"),
        n_cols=1, panel_px=420, sphere_radius=1.2, stops=_stops, value_label="this site",
        ramp=_ramp, domain=_domain, hover=False, cartoon_opacity=0.5, ligand_color="greenCarbon",
        observed_colour="#eb6834", potential_colour="#7b5cd6", offline=True)
    mo.iframe(_html, height=f"{_h + 70}px")
    return


@app.cell
def _(get_site_steps, mo, qsites, run_dir, site_dir):
    _sid = site_dir.name
    site_steps = qsites.site_status(run_dir, _sid)
    _no_clouds = " (needs clouds)" if not site_steps["clouds"] else ""
    shape_button = mo.ui.run_button(label="Shape clustering" + _no_clouds, disabled=bool(_no_clouds))
    featurise_button = mo.ui.run_button(label="Featurise" + _no_clouds, disabled=bool(_no_clouds))
    two_stage_button = mo.ui.run_button(
        label="Two-stage clustering" if site_steps["featurised"] else "Two-stage clustering (needs Featurise)",
        disabled=not site_steps["featurised"])
    _done = " · ".join(f"{name.replace('_', ' ')} {'✓' if ok else '—'}" for name, ok in site_steps.items())
    _commands = "\n".join([
        f"qpocket site shape {run_dir} --site {_sid}",
        f"qpocket site featurize {run_dir} --site {_sid}",
        f"qpocket site cluster {run_dir} --site {_sid}",
        "# or on SLURM (featurises one cloud per array task, then clusters):",
        f"qpocket slurm generate --config {run_dir / 'metadata' / 'config.yaml'} "
        f"--workdir {run_dir.parent} --site {_sid}",
    ])
    mo.vstack([
        mo.md(f"### Steps for {_sid}\n{_done}"),
        mo.hstack([shape_button, featurise_button, two_stage_button], justify="start"),
        get_site_steps() or mo.md(""),
        mo.accordion({"Run them from a terminal instead": mo.plain_text(_commands)}),
    ])
    return featurise_button, shape_button, two_stage_button


@app.cell
def _(featurise_button, mo, qsites, run_dir, set_site_steps, shape_button, site_dir, two_stage_button):
    mo.stop(not (shape_button.value or featurise_button.value or two_stage_button.value))
    import traceback as _traceback

    _name, _step = next((n, f) for n, f, b in [
        ("Shape clustering", qsites.shape_cluster_site, shape_button),
        ("Featurise", qsites.featurize_site, featurise_button),
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
def _(mo, qviz, run_dir):
    # This cell feeds the site table, so an odd config must not stop it.
    regroup_cutoff = mo.ui.number(start=0.5, stop=20.0, step=0.5,
                                  value=qviz.site_link_cutoff(run_dir, 0.5, 20.0),
                                  label="Link cutoff (Å)")
    regroup_button = mo.ui.run_button(label="Regroup sites")
    mo.accordion({"Regroup sites": mo.vstack([
        mo.callout(mo.md("Regrouping rewrites `sites/`. Every site whose ligands change loses its "
                         "per-site results: features, clustering and cross-docking."), kind="warn"),
        mo.hstack([regroup_cutoff, regroup_button], justify="start", align="end"),
        mo.plain_text(f"qpocket site regroup {run_dir} --cutoff <x>"),
    ])})
    return regroup_button, regroup_cutoff


@app.cell
def _(mo, qsites, regroup_button, regroup_cutoff, run_dir):
    # No mo.stop: the site table depends on this, and must show before any regroup.
    regrouped = None
    if regroup_button.value:
        with mo.status.spinner(title="Regrouping sites…"):
            try:
                regrouped = qsites.regroup_sites(run_dir, float(regroup_cutoff.value))
                _message = f"_Regrouped into {len(regrouped)} sites._"
            except Exception as _e:
                _message = f"_Regroup failed: {type(_e).__name__}: {_e}_"
    mo.md(_message if regroup_button.value else "")
    return (regrouped,)


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
def _(mo):
    mo.md("""
    ## Clustering
    Pick a clustering method and adjust the controls to select the best parametrization. **Save** updates the run's clustering while keeping the old one in `metadata/clustering_history/`.
    """)
    return


@app.cell
def _(mo, qviz, site_dir):
    from qpocket.arms import recluster as _recluster

    cluster_matrices = qviz.two_stage.load_matrices(site_dir)
    cluster_settings = _recluster.load_settings(site_dir)

    cluster_method = mo.ui.dropdown(
        options={label: key for key, label in qviz.two_stage.METHODS.items()},
        value=qviz.two_stage.METHODS["two_stage"],
        label="**Method**",
        full_width=True,
    )
    (cluster_method if cluster_matrices is not None
     else mo.md("_This site has no distance matrices yet; run Shape clustering above._"))
    return cluster_matrices, cluster_method, cluster_settings


@app.cell
def _(cluster_matrices, cluster_settings, info, mo, qviz):
    def _slider(lo, hi, value, label):
        return mo.ui.slider(start=lo, stop=hi, step=max((hi - lo) / 200, 1e-9),
                            value=min(max(value, lo), hi), label=label, show_value=True)

    if cluster_matrices is None:
        _combined_lo, _combined_hi, _shape_lo, _shape_hi, _shape_start = 0.0, 1.0, 0.0, 1.0, 0.5
    else:
        _combined_lo, _combined_hi = qviz.two_stage.height_range(cluster_matrices.combined)
        _shape_link = cluster_settings.shape.linkage
        _shape_lo, _shape_hi = qviz.two_stage.height_range(cluster_matrices.shape, _shape_link)
        _stored = (int(info["shape_cluster_id"].nunique())
                   if "shape_cluster_id" in info.columns else 2)
        _shape_start = (float(cluster_settings.shape.threshold)
                        if cluster_settings.shape.criterion == "threshold"
                        and cluster_settings.shape.threshold
                        else qviz.two_stage.height_for_count(cluster_matrices.shape,
                                                             max(_stored, 1), _shape_link))

    damping = mo.ui.slider(start=0.5, stop=0.98, step=0.01, value=0.9,
                           label="Damping", show_value=True)
    combined_cut = _slider(_combined_lo, _combined_hi,
                           (_combined_lo + _combined_hi) / 2, "Cut height")
    shape_cut = _slider(_shape_lo, _shape_hi, _shape_start, "Shape cut")
    min_split = mo.ui.number(start=2, stop=100, value=int(cluster_settings.min_cluster_size),
                             label="Split groups of at least")
    return combined_cut, damping, min_split, shape_cut


@app.cell
def _(cluster_matrices, cluster_settings, mo, qviz, shape_cut):
    # The chemistry range depends on where the shape cut fell, so it follows it.
    if cluster_matrices is None:
        _chem_lo, _chem_hi = 0.0, 1.0
    else:
        _groups = qviz.two_stage.shape_groups(cluster_matrices, shape_cut.value,
                                              cluster_settings.shape.linkage)
        _chem_lo, _chem_hi = qviz.two_stage.chem_height_range(cluster_matrices, _groups,
                                                              cluster_settings.chem.linkage)
    chem_cut = mo.ui.slider(start=_chem_lo, stop=_chem_hi,
                            step=max((_chem_hi - _chem_lo) / 200, 1e-9),
                            value=(_chem_lo + _chem_hi) / 2,
                            label="Chemistry cut", show_value=True)
    return (chem_cut,)


@app.cell
def _(
    chem_cut,
    cluster_matrices,
    cluster_method,
    combined_cut,
    damping,
    min_split,
    mo,
    shape_cut,
):
    # Only the knobs the chosen method uses are on screen.
    (mo.md("") if cluster_matrices is None else
     mo.vstack([mo.md("_Shape-only clustering — featurise this site for two-stage clustering._"),
                shape_cut]) if cluster_matrices.chem is None else {
        "affinity": mo.hstack([damping], justify="start", gap=2),
        "hierarchical": mo.hstack([combined_cut], justify="start", gap=2),
        "two_stage": mo.hstack([shape_cut, chem_cut, min_split], justify="start", gap=2),
    }[cluster_method.value])
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
def _(
    chem_cut,
    cluster_matrices,
    cluster_method,
    cluster_settings,
    mo,
    pending,
    qviz,
    shape_cut,
):
    # The mirrored pair only means something for the two-stage cut.
    if cluster_matrices is None or pending is None:
        _view = mo.md("")
    elif cluster_matrices.chem is None:
        _view = qviz.two_stage.dendrogram_figure(
            cluster_matrices.shape, cluster_matrices.ids, shape_cut.value, "Shape",
            cluster_settings.shape.linkage)
    elif cluster_method.value == "two_stage":
        _view = qviz.two_stage.mirrored_figure(
            cluster_matrices, pending, shape_cut.value, chem_cut.value,
            cluster_settings.shape.linkage)
    elif cluster_method.value == "hierarchical":
        _view = qviz.two_stage.dendrogram_figure(
            cluster_matrices.combined, cluster_matrices.ids, None, "Combined")
    else:
        _view = mo.md("_Affinity propagation builds no tree; see the heatmap below._")
    _view
    return


@app.cell
def _(mo, pending):
    _facts = []
    if pending is not None:
        _facts.append(f"**{pending.n_clusters} clusters**")
        if pending.silhouette is not None:
            _facts.append(f"silhouette {pending.silhouette:.2f}")
    mo.md(" · ".join(_facts) + "  \n_Preview only; press Apply to make it the run's clustering._"
          if _facts else "")
    return


@app.cell
def _(mo, pending_settings):
    apply_clustering = mo.ui.run_button(
        label="Apply to this run", disabled=pending_settings is None)
    mo.vstack([
        apply_clustering,
        mo.md("_Writes `metadata/clustering_metadata.csv` and `clustering_params.json`; "
              "the previous pair moves to `metadata/clustering_history/`._"),
    ])
    return (apply_clustering,)


@app.cell
def _(apply_clustering, mo, pending_settings, site_dir):
    # Every section below reads the file this writes, so the button is what makes
    # them refresh.
    applied_at = None
    if apply_clustering.value and pending_settings is not None:
        from qpocket.arms import recluster as _recluster_apply

        _saved = _recluster_apply.run(site_dir, pending_settings)
        applied_at = str(_saved)
    mo.md(f"Applied; `{applied_at}` is now this run's clustering."
          if applied_at else "")
    return (applied_at,)


@app.cell
def _(applied_at, dist_df, info, mo, np, pd, qviz, site_dir):
    # ---- The grouping every view below follows -------------------------------
    # The clustering section is the decision point: it writes
    # metadata/clustering_metadata.csv, and this reads that file back. Depending
    # on applied_at is what makes these views refresh after Apply.
    #
    # Silhouette is added here, since the stored metadata does not carry it. It is
    # measured on the spectral distances themselves, so a structure with a
    # negative value sits closer to a cluster other than its own.
    _ = applied_at
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
    active_info["silhouette"] = np.nan

    _mean_sil = None
    if dist_df is not None:
        _ids = [str(i) for i in dist_df.index]
        _D = dist_df.loc[dist_df.index, dist_df.index].to_numpy(dtype=float)
        _label_of = dict(zip(active_info["pdb_id"].astype(str), active_info["cluster_id"]))
        _labels = np.nan_to_num(
            np.array([_label_of.get(_i, -1) for _i in _ids], dtype=float), nan=-1
        ).astype(int)
        _mean_sil, _values = qviz.silhouette(_D, _labels)
        if _values is not None:
            active_info["silhouette"] = (
                active_info["pdb_id"].astype(str).map(dict(zip(_ids, _values)))
            )

    mo.md(
        f"**{active_info['cluster_id'].nunique()} pocket conformations** across "
        f"{len(active_info)} structures"
        + ("" if _mean_sil is None else f" · mean silhouette {_mean_sil:.2f}")
        + f" · {int(active_info['is_representative'].sum())} representatives"
    )
    return (active_info,)


@app.cell
def _(mo):
    mo.md("""
    ## Structures
    `dist_to_exemplar` — distance to the cluster representative (large = outlier) ·
    `silhouette` < 0 — fits another cluster better · `utilization` — share of the
    reference site covered.
    """)
    return


@app.cell
def _(active_info, mo):
    _cols = ["pdb_id", "cluster_id"] + [
        c
        for c in [
            "silhouette", "is_representative", "dist_to_exemplar", "utilization",
            "rmsd_to_ref", "protein_state",
        ]
        if c in active_info.columns
    ]
    mo.ui.table(active_info[_cols], selection=None, pagination=True)
    return


@app.cell
def _(dist_kind, mo):
    mo.md(f"""
    ## Pocket landscape
    One point per structure, placed by {dist_kind} distance alone. **PCA** is the honest
    default; UMAP and t-SNE separate groups, but their gaps are not distances. Lasso
    points to open their cluster below.
    """)
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
def _(active_info, dist_df, embed_method, mo, qcontrols, qviz):
    _color_opts = [
        c
        for c in ["cluster_id", "silhouette", "utilization", "protein_state", "rmsd_to_ref", "is_representative"]
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
    qcontrols.panel(
        primary=[embed_method, color_by],
        groups={embed_method.value: {"UMAP": [n_neighbors, min_dist],
                                     "t-SNE": [perplexity]}.get(embed_method.value, [])},
        label="Layout options",
    )
    return color_by, min_dist, n_neighbors, perplexity


@app.cell
def _(
    active_info,
    dist_df,
    embed_method,
    min_dist,
    mo,
    n_neighbors,
    pd,
    perplexity,
    qviz,
):
    mo.stop(dist_df is None, mo.md("No distance matrix for this site yet; run Shape clustering above."))

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
        mo.stop(True, mo.md(f"Cannot lay this run out: {_exc}"))

    emb_df = pd.DataFrame(
        {"pdb_id": _ids, "X": layout.coords[:, 0], "Y": layout.coords[:, 1]}
    ).merge(active_info, on="pdb_id", how="left")
    emb_df["cluster_id"] = emb_df["cluster_id"].astype("Int64").astype(str)
    return emb_df, layout


@app.cell
def _(color_by, dist_kind, emb_df, layout, mo, px, qviz):
    _cb = color_by.value
    _hover = [
        c
        for c in ["cluster_id", "silhouette", "utilization", "rmsd_to_ref", "protein_state", "is_representative"]
        if c in emb_df.columns
    ]

    _fig = px.scatter(
        emb_df,
        x="X",
        y="Y",
        color=_cb if _cb != "cluster_id" else emb_df["cluster_id"].astype(str),
        text="pdb_id",
        hover_name="pdb_id",
        hover_data=_hover,
        custom_data=["pdb_id"],
        # Cluster ids are categories: a continuous ramp would imply an order
        # between them, and would not match the colours in the dendrograms.
        color_discrete_map=(qviz.ramps.cluster_colours(emb_df["cluster_id"])
                            if _cb == "cluster_id" else None),
        color_continuous_scale=("Viridis" if _cb != "cluster_id"
                                and emb_df[_cb].dtype.kind in "fi" else None),
        template="plotly_white",
        height=560,
        labels={"X": layout.axis_labels[0], "Y": layout.axis_labels[1]},
    )
    _fig.update_traces(marker=dict(size=15, line=dict(width=1, color="white")), textposition="top center")
    _fig.update_layout(
        title=f"{layout.method} of {dist_kind} distances · coloured by {_cb}",
        dragmode="lasso",
    )

    embed_plot = mo.ui.plotly(_fig)
    mo.vstack([embed_plot, mo.md(f"_{layout.note}_")])
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
def _(mo):
    mo.md("""
    ## Pockets side by side
    Panels share framing and colour limits, so colour differences are real. Drag one
    to rotate all.
    """)
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
def _(FEATURE_COLS, active_info, mo, panel_mode, qcontrols, umap_selection):
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
    qcontrols.panel(
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
    )
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
def _(
    Path,
    active_info,
    cartoon_opacity,
    clip_pct,
    cluster_sel,
    color_scheme,
    colour_mode,
    context_radius,
    feature,
    cloud_path,
    grid_spacing,
    hover_values,
    json,
    ligand_color,
    load_pocket,
    mo,
    n_cols,
    np,
    observed_colour,
    offline_viewer,
    panel_mode,
    panel_px,
    pd,
    potential_colour,
    protein_style,
    qviz,
    residue_picker,
    run_interactions,
    show_all_sidechains,
    show_ligand,
    show_pocket,
    sphere_radius,
):
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
        mo.stop(cluster_sel.value is None, mo.md("No cluster available."))
        _fallbacks = []
        _rows = active_info[active_info["cluster_id"].astype(str) == str(cluster_sel.value)]

    # ---- Gather the pockets of this cluster ---------------------------------
    pockets = []  # (pdb_id, metadata row, point cloud)
    missing = []
    for _, _row in _rows.iterrows():
        _fp = cloud_path(_row["pdb_id"])
        if _fp.exists():
            pockets.append((str(_row["pdb_id"]), _row, load_pocket(str(_fp))))
        else:
            missing.append(str(_row["pdb_id"]))

    mo.stop(not pockets, mo.md(f"No pocket cloud found for cluster {cluster_sel.value}."))

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
            (r for _, r, _ in sorted(pockets, key=lambda t: -_sizes.get(t[1]["cluster_id"], 0))),
            None,
        )
    else:
        _rep_row = next((r for _, r, _ in pockets if bool(r.get("is_representative"))), None)
    if mode != "feature value" and grid_spacing is None:
        mode = "feature value"
        mo.output.append(
            mo.md("_No common lattice for these pockets, so showing feature values._")
        )
    if mode == "difference from representative" and _rep_row is None:
        mode = "feature value"
        mo.output.append(mo.md("_This cluster has no representative, so showing feature values._"))

    panel_values, value_label, ramp, domain, stops = qviz.grid.panel_values(
        pockets, mode, feat, _rep_row, grid_spacing, color_scheme.value, float(clip_pct.value))

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
    mo.vstack(
        [
            mo.iframe(_html, height=f"{grid_h + 70}px"),
            mo.md("_" + " · ".join(_notes) + "_"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Ligands
    One depiction per distinct molecule, most frequent first. Filter by cluster to
    link chemotypes to conformations.
    """)
    return


@app.cell
def _(actives_df, mo, pd):
    # ---- Canonical SMILES, computed once per run ----------------------------
    # RDKit parsing is the slow part and a screening table can run to a few
    # thousand rows, so it happens here rather than in the cell the cluster
    # selector re-runs.
    mo.stop(
        actives_df is None or actives_df.empty,
        mo.md("No `screening/actives.smi` for this site, so there is no ligand chemistry to draw."),
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
def _(active_info, mo, qcontrols):
    _clusters = [str(c) for c in sorted(active_info["cluster_id"].dropna().unique())]
    ligand_cluster = mo.ui.dropdown(
        options=["all clusters", *_clusters], value="all clusters", label="Cluster"
    )
    ligand_cols = mo.ui.slider(start=1, stop=6, value=4, label="Columns")
    # A screening table can name hundreds of compounds and every panel is drawn as
    # SVG paths, so the cap is what keeps a large run from rendering megabytes.
    max_ligands = mo.ui.slider(start=4, stop=120, step=4, value=24, label="Max molecules")
    show_ligand_props = mo.ui.checkbox(value=True, label="Properties table")
    qcontrols.panel(
        primary=[ligand_cluster],
        groups={"Layout": [ligand_cols, max_ligands],
                "Include": [show_ligand_props]},
    )
    return ligand_cluster, ligand_cols, max_ligands, show_ligand_props


@app.cell
def _(
    active_info,
    ligand_cluster,
    ligand_cols,
    ligand_rows,
    max_ligands,
    mo,
    n_unparsed,
    qviz,
    show_ligand_props,
):
    # The table can reach beyond the structures of this run, and a cluster is a
    # subset of those, so both are filtered on the active grouping.
    _rows, _outside, _cluster_of = qviz.ligands.in_scope(ligand_rows, active_info, ligand_cluster.value)
    mo.stop(
        _rows.empty,
        mo.md(f"_No structure in cluster {ligand_cluster.value} was solved with a ligand._"),
    )
    _unique = qviz.ligands.distinct_molecules(_rows, _cluster_of)
    _shown = _unique[: int(max_ligands.value)]
    _svg, _mols = qviz.ligands.depictions(_shown, n_cols=int(ligand_cols.value),
                                          all_clusters=ligand_cluster.value == "all clusters")
    _notes = qviz.ligands.notes(_shown, _unique, _rows, _outside, n_unparsed)
    mo.vstack(
        [
            mo.Html(f'<div style="overflow-x:auto">{_svg}</div>'),
            mo.md("_" + " · ".join(_notes) + "_"),
        ]
        + ([mo.ui.table(qviz.ligands.properties(_shown, _mols), selection=None, pagination=True)]
           if show_ligand_props.value else [])
    )
    return

@app.cell
def _(mo):
    mo.md("""
    ## Distance matrix
    Similar pockets form blocks on the diagonal; a dark row inside a block is a
    structure worth checking.
    """)
    return


@app.cell
def _(mo):
    dist_order = mo.ui.dropdown(
        options=["cluster blocks", "hierarchical", "as loaded"],
        value="cluster blocks",
        label="Row / column order",
    )
    dist_order
    return (dist_order,)


@app.cell
def _(active_info, dist_df, dist_order, mo, qviz):
    mo.stop(dist_df is None, mo.md("No distance matrix."))
    _M = qviz.matrix.symmetrised(dist_df)
    _order, _blocks = qviz.matrix.ordering(_M, active_info, dist_order.value)
    qviz.matrix.heatmap(_M, _order, _blocks, dist_order.value)
    return

@app.cell
def _(mo):
    mo.md("""
    ## Interaction fingerprint
    What the ligands touch, and what merely lines the pocket. Percentages on the
    left are over the ligand-bound structures; on the right, the pale bar is over
    every structure, since an apo pocket still has a lining. Rows are the same in
    both panels, and in every cluster.
    """)
    return


@app.cell
def _(active_info, mo, qcontrols, run_interactions):
    mo.stop(run_interactions is None,
            mo.md("No `observed_interactions.h5` or `potential_interactions.h5` in this run."))

    _clusters = [str(c) for c in sorted(active_info["cluster_id"].dropna().unique())]
    fingerprint_scope = mo.ui.dropdown(
        options=["all clusters", *_clusters], value="all clusters", label="**Show**")
    potential_mode = mo.ui.dropdown(
        options=["every pocket residue", "never contacted"],
        value="every pocket residue", label="Right panel")
    min_frequency = mo.ui.slider(start=0, stop=100, step=5, value=0,
                                 label="Hide below (%)", show_value=True)
    row_px = mo.ui.slider(start=12, stop=40, step=2, value=24, label="Row height (px)")

    qcontrols.panel(
        primary=[fingerprint_scope],
        groups={"Rows": [potential_mode, min_frequency, row_px]},
    )
    return fingerprint_scope, min_frequency, potential_mode, row_px


@app.cell
def _(
    active_info,
    fingerprint_scope,
    min_frequency,
    mo,
    potential_mode,
    qviz,
    row_px,
    run_interactions,
):
    mo.stop(run_interactions is None, mo.md(""))

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
            mo.md(f"_Nothing left to plot for {_label} at this cutoff._"))

    fingerprint_plot = mo.ui.plotly(
        qviz.interactions.observed_figure(_freqs, height_per_row=int(row_px.value)))
    _right = mo.ui.plotly(
        qviz.interactions.potential_figure(_freqs, height_per_row=int(row_px.value)))

    _notes = [f"{len(_freqs.residues)} residues",
              f"{_freqs.n_holo} of {_freqs.n_members} structures carry a ligand"]
    if run_interactions.folded:
        _notes.append(f"folded into \u201cOther\u201d: {', '.join(run_interactions.folded)}")

    mo.vstack([
        mo.hstack([fingerprint_plot, _right], widths=[1, 1], gap=0.5),
        mo.md("_" + " \u00b7 ".join(_notes) + "_"),
    ])
    return (fingerprint_plot,)


@app.cell
def _(fingerprint_plot, mo, run_interactions):
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
    mo.hstack([residue_picker], justify="start")
    return (residue_picker,)


@app.cell
def _(mo):
    mo.md("""
    ## Can two states be targeted at once?
    For every pair of clusters: how far their engaged residues overlap, how far
    their pockets overlap in space, and whether a shared residue changes its
    chemistry between them — a donor in one state and an acceptor in the other is
    a contradiction, not a difference. A pair resting on too few structures is
    left blank rather than called.
    """)
    return


@app.cell
def _(active_info, cloud_path, grid_spacing, load_pocket, mo, qviz, run_interactions):
    mo.stop(run_interactions is None, mo.md("No interaction archives in this run."))

    _holo = set(run_interactions.holo)
    exclusivity_members = {}
    for _cluster, _rows in active_info.groupby("cluster_id", sort=True):
        _ids = [p for p in _rows["pdb_id"].astype(str) if p in _holo]
        if _ids:
            exclusivity_members[int(_cluster)] = _ids

    mo.stop(len(exclusivity_members) < 2,
            mo.md("_Fewer than two clusters contain a ligand-bound structure, so there is "
                  "no pair to compare._"))

    # Shape is measured over the member clouds, not between representatives: one
    # structure should not decide whether a whole state is reachable.
    _shape = None
    if grid_spacing is not None:
        _clouds = {
            _pid: load_pocket(str(cloud_path(_pid)))[["x", "y", "z"]].to_numpy(dtype=float)
            for _ids in exclusivity_members.values() for _pid in _ids
            if cloud_path(_pid).exists()
        }
        if _clouds:
            _shape = qviz.interactions.shape_overlaps(_clouds, exclusivity_members, grid_spacing)

    exclusivity_profile = qviz.interactions.observed_profile(run_interactions)
    verdicts = qviz.interactions.pair_verdicts(exclusivity_profile, exclusivity_members,
                                               shape=_shape)

    mo.vstack([
        mo.ui.plotly(qviz.interactions.exclusivity_figure(
            verdicts, sorted(exclusivity_members))),
        mo.md("_" + ("pocket shape included in the verdict" if _shape
                     else "no common lattice, so shape is left out of the verdict")
              + f" \u00b7 {len(verdicts)} pairs_"),
    ])
    return exclusivity_members, exclusivity_profile, verdicts


@app.cell
def _(mo, verdicts):
    _pairs = {f"cluster {v.a} vs {v.b}": f"{v.a}-{v.b}" for v in verdicts}
    pair_choice = mo.ui.dropdown(
        options=_pairs, value=next(iter(_pairs), None), label="**Pair**")
    pair_choice
    return (pair_choice,)


@app.cell
def _(exclusivity_members, exclusivity_profile, mo, pair_choice, qviz, verdicts):
    mo.stop(pair_choice.value is None, mo.md(""))

    from qpocket.core.interactions import residue_enrichment as _residue_enrichment

    _a, _b = (int(part) for part in str(pair_choice.value).split("-"))
    _verdict = next(v for v in verdicts if (v.a, v.b) == (_a, _b))
    _tests = _residue_enrichment(exclusivity_profile,
                                 exclusivity_members[_a], exclusivity_members[_b])

    _core = ", ".join(_verdict.shared[:8]) or "none"
    _more = len(_verdict.shared) - 8
    mo.vstack([
        mo.md(f"**{qviz.interactions.describe(_verdict)}**"),
        mo.ui.plotly(qviz.interactions.pair_figure(_tests, _verdict)),
        mo.md(f"_Shared core: {_core}{f' +{_more}' if _more > 0 else ''} \u00b7 "
              f"only in cluster {_a}: {', '.join(_verdict.a_only) or 'none'} \u00b7 "
              f"only in cluster {_b}: {', '.join(_verdict.b_only) or 'none'}_"),
    ])
    return


@app.cell
def _(mo):
    from qpocket.arms.screen import crossdock

    crossdock_prepare = mo.ui.run_button(label="Prepare cross-docking")
    crossdock_analyse = mo.ui.run_button(label="Analyse cross-docking")
    mo.vstack([
        mo.md("## Screening: cross-docking\n\nEvery distinct active of this site into every cluster representative. "
              "Prepare writes a task folder you can run here or copy to a cluster; Analyse reads the results."),
        mo.hstack([crossdock_prepare, crossdock_analyse], justify="start"),
    ])
    return crossdock, crossdock_analyse, crossdock_prepare


@app.cell
def _(crossdock, crossdock_prepare, mo, site_dir):
    mo.stop(not crossdock_prepare.value)

    try:
        _plan = crossdock.prepare(site_dir)
        _out = mo.vstack([mo.md(f"Prepared `{_plan.jobs_csv}`. Run the dockings with:"),
                          mo.plain_text(_plan.command)])
    except Exception as _e:
        _out = mo.md(f"_Preparation failed: {type(_e).__name__}: {_e}_")
    _out
    return


@app.cell
def _(crossdock, crossdock_analyse, mo, qviz, site_dir):
    mo.stop(not crossdock_analyse.value)
    from qpocket.screen_config import load_screen_config as _load

    try:
        _res = crossdock.analyze(site_dir)
        _threshold = _load().analysis.rmsd_threshold
        _out = mo.vstack([
            mo.ui.table(_res.success, selection=None),
            qviz.crossdock.rmsd_heatmap(_res.pairs, _threshold) if len(_res.pairs) else mo.md("_No poses yet._"),
        ])
    except FileNotFoundError as _e:
        _out = mo.md(f"_Nothing to analyse yet: {_e}_")
    _out
    return


if __name__ == "__main__":
    app.run()
