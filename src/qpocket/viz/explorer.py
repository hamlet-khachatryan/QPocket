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
    from qpocket.viz import controls as qcontrols

    return Path, json, lru_cache, mo, np, pd, px, qcontrols, qviz


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
def _(Path, mo, pd, run_selector):
    mo.stop(run_selector.value is None, mo.md("No run selected / none found."))

    run_dir = Path(run_selector.value)
    meta = run_dir / "metadata"
    pockets_dir = run_dir / "pockets"

    def _read_csv(p, **kw):
        return pd.read_csv(p, **kw) if p.exists() else None

    master_df = _read_csv(meta / "master.csv")
    clustering_df = _read_csv(meta / "clustering_metadata.csv")
    util_df = _read_csv(meta / "binding_pocket_utilization.csv")


    actives_df = _read_csv(run_dir / "screening" / "actives.smi", sep="\t")

    dist_path = meta / "spectral_distance_matrix.csv"
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

    def featurized_path(pdb_id: str) -> Path:
        return pockets_dir / str(pdb_id) / "featurized.csv"

    #available features
    FEATURE_COLS = []
    for _pid in info["pdb_id"]:
        _fp = featurized_path(_pid)
        if _fp.exists():
            _cols = pd.read_csv(_fp, nrows=1).columns.tolist()
            FEATURE_COLS = [c for c in _cols if c not in ("x", "y", "z")]
            break

    mo.md(
        f"""
        `{run_dir}` — structures: **{len(info)}** · clusters:
        **{info['cluster_id'].nunique()}** · representatives:
        **{int(info['is_representative'].sum())}** · features:
        `{', '.join(FEATURE_COLS) or 'none found'}`
        """
    )
    return (
        FEATURE_COLS,
        actives_df,
        dist_df,
        featurized_path,
        info,
        run_dir,
    )


@app.cell
def _(lru_cache, pd):
    @lru_cache(maxsize=512)
    def load_pocket(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    return (load_pocket,)


@app.cell
def _(qviz, run_dir):
    # Read once, in a cell of its own: both the 3D panels and the fingerprint
    # section below depend on this, and marimo resolves cells by dataflow, so a
    # shared upstream cell is what keeps them from forming a cycle.
    run_interactions = qviz.interactions.load_run_interactions(run_dir)
    return (run_interactions,)


@app.cell
def _(mo):
    mo.md("""
    ## Clustering
    Pick a clustering method and adjust the controls to select the best parametrization. **Save** updates the run's clustering while keeping the old one in `metadata/clustering_history/`.
    """)
    return


@app.cell
def _(mo, qviz, run_dir):
    from qpocket.pipelines import recluster as _recluster

    cluster_matrices = qviz.two_stage.load_matrices(run_dir)
    cluster_settings = _recluster.load_settings(run_dir)

    cluster_method = mo.ui.dropdown(
        options={label: key for key, label in qviz.two_stage.METHODS.items()},
        value=qviz.two_stage.METHODS["two_stage"],
        label="**Method**",
        full_width=True,
    )
    (cluster_method if cluster_matrices is not None
     else mo.md("_This run predates the shape and chemistry matrices, so it is "
                "shown with the clustering it was saved with._"))
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
    (mo.md("") if cluster_matrices is None else {
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
        pending_settings = qviz.two_stage.settings_for(
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
def _(apply_clustering, mo, pending_settings, run_dir):
    # Every section below reads the file this writes, so the button is what makes
    # them refresh.
    applied_at = None
    if apply_clustering.value and pending_settings is not None:
        from qpocket.pipelines import recluster as _recluster_apply

        _saved = _recluster_apply.run(run_dir, pending_settings)
        applied_at = str(_saved)
    mo.md(f"Applied; `{applied_at}` is now this run's clustering."
          if applied_at else "")
    return (applied_at,)


@app.cell
def _(applied_at, dist_df, info, mo, np, pd, qviz, run_dir):
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
    _stored_path = run_dir / "metadata" / "clustering_metadata.csv"
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
def _(mo):
    mo.md("""
    ## Pocket landscape
    One point per structure, placed by spectral distance alone. **PCA** is the honest
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
    mo.stop(dist_df is None, mo.md("No `spectral_distance_matrix.csv` for this run."))

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
def _(color_by, emb_df, layout, mo, px, qviz):
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
        title=f"{layout.method} of spectral distances · coloured by {_cb}",
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
def _(active_info, featurized_path, load_pocket, np, qviz):
    # ---- The lattice the viewer's set-arithmetic modes rest on ---------------
    # Every pocket in a run is sampled on the same regular lattice, which is what
    # makes "difference from representative" and "consensus occupancy" exact rather
    # than interpolated. Never fatal: a run whose spacing cannot be established
    # still gets a viewer, just without those two modes.
    _clouds = [
        load_pocket(str(featurized_path(_pid)))[["x", "y", "z"]].to_numpy(dtype=float)
        for _pid in active_info["pdb_id"]
        if featurized_path(_pid).exists()
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
    featurized_path,
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
    mo.stop(feature.value is None, mo.md("No feature available."))

    CONTACT_R = 4.0  # side chains are drawn for residues this close to the cloud

    feat = feature.value
    across_clusters = panel_mode.value != "one cluster"
    _sizes = active_info["cluster_id"].value_counts().to_dict()

    if across_clusters:
        # One panel per cluster, showing the structure that stands for it: the
        # method's own exemplar where there is one, otherwise the member closest to
        # its cluster's exemplar, so no cluster is silently left out.
        _picked, _fallbacks = [], []
        for _cluster, _members in active_info.groupby("cluster_id", sort=True):
            _reps = _members[_members["is_representative"].fillna(False)]
            if len(_reps):
                _picked.append(_reps.iloc[0])
                continue
            _fallbacks.append(int(_cluster))
            if "dist_to_exemplar" in _members.columns and _members["dist_to_exemplar"].notna().any():
                _picked.append(_members.nsmallest(1, "dist_to_exemplar").iloc[0])
            else:
                _picked.append(_members.iloc[0])
        _rows = pd.DataFrame(_picked)
    else:
        mo.stop(cluster_sel.value is None, mo.md("No cluster available."))
        _fallbacks = []
        _rows = active_info[active_info["cluster_id"].astype(str) == str(cluster_sel.value)]

    # ---- Gather the pockets of this cluster ---------------------------------
    pockets = []  # (pdb_id, metadata row, point cloud)
    missing = []
    for _, _row in _rows.iterrows():
        _fp = featurized_path(_row["pdb_id"])
        if _fp.exists():
            pockets.append((str(_row["pdb_id"]), _row, load_pocket(str(_fp))))
        else:
            missing.append(str(_row["pdb_id"]))

    mo.stop(not pockets, mo.md(f"No `featurized.csv` found for cluster {cluster_sel.value}."))

    # ---- What the spheres encode, and on what scale --------------------------
    # Each mode produces (coordinates, values) per panel. The difference and
    # consensus modes are set operations on the shared lattice, so they need no
    # interpolation; a site only one pocket occupies has no difference to show and
    # is dropped rather than filled in.
    mode = colour_mode.value
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

    panel_values = {}   # pdb_id -> (coords, values)
    if mode == "consensus occupancy":
        _occ = qviz.occupancy([p[["x", "y", "z"]].to_numpy(dtype=float) for _, _, p in pockets], grid_spacing)
        for pid, _, pocket in pockets:
            _own = qviz.voxel_index(pocket[["x", "y", "z"]].to_numpy(dtype=float), grid_spacing)
            _own_keys = {tuple(_r) for _r in _own}
            _keep = np.array(
                [tuple(_r) in _own_keys for _r in qviz.voxel_index(_occ.coords, grid_spacing)],
                dtype=bool,
            )
            panel_values[pid] = (_occ.coords[_keep], _occ.fraction[_keep])
        value_label = f"in how many of {_occ.n_pockets} pockets"
        ramp = qviz.RampSpec("sequential", "Viridis", ("this pocket only", "every pocket"))
        domain = qviz.ColourDomain(0.0, 1.0, None)
    elif mode == "difference from representative":
        _ref_id = str(_rep_row["pdb_id"])
        _ref = next(p for pid, _, p in pockets if pid == _ref_id)
        _ref_xyz = _ref[["x", "y", "z"]].to_numpy(dtype=float)
        _deltas = []
        for pid, _, pocket in pockets:
            _diff = qviz.field_difference(
                _ref_xyz, _ref[feat].to_numpy(dtype=float),
                pocket[["x", "y", "z"]].to_numpy(dtype=float), pocket[feat].to_numpy(dtype=float),
                grid_spacing,
            )
            panel_values[pid] = (_diff.coords, _diff.delta)
            _deltas.append(_diff.delta)
        value_label = f"{feat} minus {_ref_id}"
        ramp = qviz.RampSpec("diverging", qviz.feature_ramp(feat, color_scheme.value).scale,
                             (f"lower than {_ref_id}", f"higher than {_ref_id}"))
        domain = qviz.colour_domain(np.concatenate(_deltas) if _deltas else [0.0],
                                    float(clip_pct.value), "diverging")
    else:
        for pid, _, pocket in pockets:
            panel_values[pid] = (
                pocket[["x", "y", "z"]].to_numpy(dtype=float),
                pocket[feat].to_numpy(dtype=float) if feat in pocket.columns else np.zeros(len(pocket)),
            )
        value_label = feat
        ramp = qviz.feature_ramp(feat, color_scheme.value)
        domain = qviz.colour_domain(
            np.concatenate([v for _, v in panel_values.values()]), float(clip_pct.value), ramp.kind
        )
    stops = qviz.ramp_stops(ramp.scale, domain.zero_frac)

    # ---- Shared framing box (pockets are pre-aligned) -----------------------
    _pts = np.vstack([p[["x", "y", "z"]].to_numpy(dtype=float) for _, _, p in pockets])
    bbox_pdb = qviz.pseudo_atom_pdb(qviz.bbox_corners(_pts), resname="BOX", chain="Z")

    # ---- Per-panel molecular models -----------------------------------------
    panels = []
    for pid, row, pocket in pockets:
        _xyz, _vals = panel_values[pid]

        _cloud = ""
        if show_pocket.value and len(_xyz):
            _cloud = qviz.pseudo_atom_pdb(_xyz, b=qviz.scale_to_percent(_vals, domain))

        _models = qviz.ProteinModels("", "", 0, 0, 0)
        _highlights = {}
        _aligned = row.get("aligned_file")
        _has_structure = isinstance(_aligned, str) and Path(_aligned).exists()

        if _has_structure and residue_picker.value:
            # Picked residues are drawn whatever the context radius is, each class
            # in its own model so it can carry its own colour scheme.
            _groups = (qviz.interactions.classify_residues(run_interactions, residue_picker.value)
                       if run_interactions is not None
                       else {"observed": tuple(residue_picker.value)})
            _highlights = qviz.residue_models(_aligned, _groups)

        if protein_style.value != "none" and _has_structure:
            # Every residue that interacts anywhere in the run, in every panel: an
            # absence then means the conformation moved it, not that a radius
            # filtered it out.
            _always = (sorted(run_interactions.engaged)
                       if run_interactions is not None and show_all_sidechains.value else ())
            _models = qviz.pocket_context_models(
                _aligned,
                pocket[["x", "y", "z"]].to_numpy(dtype=float),
                float(context_radius.value),
                CONTACT_R,
                always=_always,
            )
            if protein_style.value == "cartoon only":
                _models = _models._replace(sidechains="")

        _lig = ""
        _lp = row.get("ligand_file")
        if show_ligand.value and isinstance(_lp, str) and _lp and Path(_lp).exists():
            _lig = qviz.ligand_pdb(_lp)

        _title = []
        if across_clusters:
            _title.append(f"cluster {int(row['cluster_id'])} (n={_sizes.get(row['cluster_id'], 0)})")
        _title.append(pid)
        if bool(row.get("is_representative")) and not across_clusters:
            _title.append("rep")
        if len(_vals):
            _title.append(f"mean {float(np.mean(_vals)):.2f}")
        else:
            _title.append("no shared sites")

        panels.append(
            {
                "id": pid,
                "title": " · ".join(_title),
                "cloud": _cloud,
                "cartoon": _models.cartoon,
                "side": _models.sidechains,
                "highlights": _highlights,
                "n_forced": int(_models.n_forced),
                "ligand": _lig,
                "vals": [round(float(v), 3) for v in _vals],
                "n_points": int(len(_xyz)),
                "n_ctx": int(_models.n_context),
            }
        )

    ncol = max(1, min(int(n_cols.value), len(panels)))
    nrow = int(np.ceil(len(panels) / ncol))
    grid_h = int(panel_px.value) * nrow

    payload = {
        "rows": nrow,
        "cols": ncol,
        "panels": panels,
        "bbox": bbox_pdb,
        "radius": float(sphere_radius.value),
        "colorscheme": qviz.gradient_spec(stops),
        "feature": value_label,
        "hover": bool(hover_values.value),
        # One opacity for the whole cartoon model: 3Dmol splits cartoon geometry
        # wherever adjacent residues disagree on opacity, so keep it uniform.
        "cartoonOpacity": float(cartoon_opacity.value),
        "ligandScheme": ligand_color.value,
        "highlightSchemes": {
            "observed": qviz.interactions.carbon_scheme(observed_colour.value),
            "potential": qviz.interactions.carbon_scheme(potential_colour.value),
        },
    }

    # ---- One canvas, one WebGL context, one viewer per cell -----------------
    # createViewerGrid splits a single canvas into viewports, so the panel count
    # is not capped by the browser's WebGL context limit, and control_all makes a
    # drag in any panel drive every panel.
    _caps = "".join(
        f'<div class="cap" style="left:{(i % ncol) / ncol * 100:.4f}%;'
        f'top:{(i // ncol) / nrow * 100:.4f}%;width:{100 / ncol:.4f}%">{p["title"]}</div>'
        for i, p in enumerate(panels)
    )
    _legend_css = ",".join(qviz.legend_gradient(stops))
    # Mark where zero falls, but only when it is actually inside the domain.
    _zero_tick = ""
    if domain.zero_frac is not None and 0.02 < domain.zero_frac < 0.98:
        _zero_tick = (
            f'<div id="zero" style="left:{domain.zero_frac * 100:.2f}%"></div>'
            f'<div id="zerolab" style="left:{domain.zero_frac * 100:.2f}%">0</div>'
        )
    _html = f"""
    <style>
      body {{ margin:0; font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#0b0b0b; background:#fff; }}
      #wrap {{ position:relative; width:100%; height:{grid_h}px; }}
      #grid {{ width:100%; height:{grid_h}px; }}
      .cap {{ position:absolute; pointer-events:none; text-align:center; padding-top:4px;
             font-size:12px; color:#52514e; z-index:2; }}
      #bar {{ display:flex; align-items:center; gap:8px; padding:8px 4px 2px; color:#52514e; }}
      #rampbox {{ position:relative; flex:0 0 300px; }}
      #ramp {{ height:12px; border:1px solid #d8d7d2; border-radius:2px;
              background:linear-gradient(to right,{_legend_css}); }}
      #zero {{ position:absolute; top:-3px; width:1px; height:18px; background:#1a1a19; }}
      #zerolab {{ position:absolute; top:15px; transform:translateX(-50%); font-size:10px; color:#52514e; }}
      .end {{ font-size:11px; }}
      #err {{ display:none; padding:12px; color:#8a1f11; }}
    </style>
    <div id="wrap"><div id="grid"></div>{_caps}</div>
    <div id="bar">
      <span><b>{value_label}</b></span>
      <span class="end">{ramp.ends[0]}</span><span>{domain.lo:.2f}</span>
      <div id="rampbox"><div id="ramp"></div>{_zero_tick}</div>
      <span>{domain.hi:.2f}</span><span class="end">{ramp.ends[1]}</span>
      <span style="margin-left:10px">{ramp.scale} · drag any panel, all follow</span>
    </div>
    <div id="err">3Dmol.js did not load. Tick “offline viewer” to inline the vendored build.</div>
    {qviz.threedmol_script(offline=bool(offline_viewer.value))}
    <script>
    const D = {json.dumps(payload)};
    function build() {{
      if (typeof $3Dmol === "undefined") {{ document.getElementById("err").style.display = "block"; return; }}
      const viewers = $3Dmol.createViewerGrid(
        document.getElementById("grid"),
        {{ rows: D.rows, cols: D.cols, control_all: true }},
        {{ backgroundColor: "white", antialias: true }}
      );
      const all = [];
      D.panels.forEach((p, i) => {{
        const v = viewers[Math.floor(i / D.cols)][i % D.cols];
        all.push(v);
        // Invisible box first: every panel frames the identical volume.
        const box = v.addModel(D.bbox, "pdb", {{ assignBonds: false }});
        if (p.cartoon) {{
          const m = v.addModel(p.cartoon, "pdb");
          v.setStyle({{ model: m }}, {{ cartoon: {{ color: "#b9bdc4", thickness: 0.35, opacity: D.cartoonOpacity }} }});
        }}
        if (p.side) {{
          const m = v.addModel(p.side, "pdb");
          v.setStyle({{ model: m }}, {{ stick: {{ radius: 0.11 }} }});
        }}
        Object.entries(p.highlights || {{}}).forEach(([cls, pdb]) => {{
          const m = v.addModel(pdb, "pdb");
          const scheme = D.highlightSchemes[cls];
          v.setStyle({{ model: m }}, {{ stick: {{ radius: 0.22, colorscheme: scheme }} }});
          v.addStyle({{ model: m }}, {{ sphere: {{ radius: 0.28, colorscheme: scheme }} }});
        }});
        if (p.ligand) {{
          const m = v.addModel(p.ligand, "pdb");
          v.setStyle({{ model: m }}, {{
            stick: {{ radius: 0.17, colorscheme: D.ligandScheme }},
            sphere: {{ radius: 0.30, colorscheme: D.ligandScheme }},
          }});
        }}
        if (p.cloud) {{
          const m = v.addModel(p.cloud, "pdb", {{ assignBonds: false }});
          v.setStyle({{ model: m }}, {{ sphere: {{ radius: D.radius, colorscheme: D.colorscheme }} }});
          if (D.hover) v.setHoverable({{ model: m }}, true,
            (atom, viewer) => {{
              if (atom.label) return;
              const val = p.vals[atom.serial - 1];
              atom.label = viewer.addLabel(
                p.id + "  " + D.feature + " " + (val === undefined ? "?" : val),
                {{ position: atom, backgroundColor: "#1a1a19", backgroundOpacity: 0.85,
                   fontColor: "white", fontSize: 11 }}
              );
              viewer.render();
            }},
            (atom, viewer) => {{
              if (atom.label) {{ viewer.removeLabel(atom.label); delete atom.label; viewer.render(); }}
            }}
          );
        }}
        v.zoomTo({{ model: box }});
      }});
      all.forEach(v => v.render());
      window.addEventListener("resize", () => all.forEach(v => {{ v.resize(); v.render(); }}));
    }}
    if (document.readyState === "loading") {{
      document.addEventListener("DOMContentLoaded", build);
    }} else {{
      build();
    }}
    </script>
    """

    _notes = []
    if across_clusters:
        _notes.append(f"one representative from each of {len(panels)} clusters")
    if _fallbacks:
        _notes.append(
            f"no marked representative for cluster(s) {', '.join(str(c) for c in _fallbacks)};"
            " used the member closest to the exemplar"
        )
    if missing:
        _notes.append(f"No `featurized.csv` for: {', '.join(missing)}")
    if protein_style.value != "none":
        _note = (f"protein: residues within {context_radius.value} Å of the cloud "
                 f"(median {int(np.median([p['n_ctx'] for p in panels]))} per panel); "
                 f"side chains within {CONTACT_R:.0f} Å")
        _forced = [p["n_forced"] for p in panels]
        if any(_forced):
            _note += (f", plus every interacting residue "
                      f"({min(_forced)}–{max(_forced)} of "
                      f"{len(run_interactions.engaged)} found per panel)")
        _notes.append(_note)
    _notes.append(f"{sum(p['n_points'] for p in panels)} spheres across {len(panels)} panels")
    if mode != "feature value":
        _notes.append(f"spheres show {value_label}")
    if residue_picker.value:
        _classes = (", ".join(
            f"{len(v)} {k}" for k, v in qviz.interactions.classify_residues(
                run_interactions, residue_picker.value).items())
            if run_interactions is not None else str(len(residue_picker.value)))
        _notes.append(f"highlighted: {_classes}")

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
        mo.md("No `screening/actives.smi` in this run, so there is no ligand chemistry to draw."),
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
    pd,
    show_ligand_props,
):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    # ---- Which structures a depiction may come from -------------------------
    # The table can reach beyond the structures of this run, and a cluster is a
    # subset of those, so both are filtered on the active grouping.
    _cluster_of = {
        str(_row.pdb_id).upper(): int(_row.cluster_id)
        for _row in active_info.itertuples()
        if pd.notna(_row.cluster_id)
    }
    _rows = ligand_rows[ligand_rows["pdb_id"].isin(_cluster_of)]
    _outside = len(ligand_rows) - len(_rows)
    if ligand_cluster.value != "all clusters":
        _wanted = str(ligand_cluster.value)
        _rows = _rows[_rows["pdb_id"].map(lambda p: str(_cluster_of[p]) == _wanted)]

    mo.stop(
        _rows.empty,
        mo.md(f"_No structure in cluster {ligand_cluster.value} was solved with a ligand._"),
    )

    # ---- One entry per distinct molecule ------------------------------------
    _unique = []
    for _canonical, _group in _rows.groupby("canonical", sort=False):
        _pdb_ids = list(dict.fromkeys(_group["pdb_id"]))
        _unique.append(
            {
                "canonical": _canonical,
                "codes": list(dict.fromkeys(_group["code"])),
                "pdb_ids": _pdb_ids,
                "clusters": sorted({_cluster_of[_p] for _p in _pdb_ids}),
                "poses": len(_group),
            }
        )
    # Most widely crystallised first: those are the chemotypes the site really binds.
    _unique.sort(key=lambda u: (-len(u["pdb_ids"]), u["codes"][0]))
    _shown = _unique[: int(max_ligands.value)]

    # ---- Draw them as one SVG grid ------------------------------------------
    rdDepictor.SetPreferCoordGen(True)  # cleaner ring layouts than the default
    _mols, _legends = [], []
    for _u in _shown:
        _mol = Chem.MolFromSmiles(_u["canonical"])
        rdDepictor.Compute2DCoords(_mol)
        _mols.append(_mol)
        _more = len(_u["pdb_ids"]) - 3
        # ASCII only, and pipes rather than the middle dot used elsewhere: RDKit
        # draws legends in its own font, which has no glyph for "·".
        _caption = [
            "/".join(_u["codes"]),
            ", ".join(_u["pdb_ids"][:3]) + (f" +{_more}" if _more > 0 else ""),
        ]
        if ligand_cluster.value == "all clusters":
            _caption.append(f"cluster {', '.join(str(_c) for _c in _u['clusters'])}")
        _legends.append(" | ".join(_caption))

    _ncol = max(1, min(int(ligand_cols.value), len(_mols)))
    _nrow = -(-len(_mols) // _ncol)
    _w, _h = 300, 265
    _drawer = rdMolDraw2D.MolDraw2DSVG(_w * _ncol, _h * _nrow, _w, _h)
    _opts = _drawer.drawOptions()
    _opts.legendFontSize = 15
    # Per-panel scaling: one large molecule in the set would otherwise shrink
    # every other depiction to match it.
    _opts.drawMolsSameScale = False
    _drawer.DrawMolecules(_mols, legends=_legends)
    _drawer.FinishDrawing()
    _svg = _drawer.GetDrawingText()
    _svg = _svg[_svg.index("<svg"):] if "<svg" in _svg else _svg

    _props = pd.DataFrame(
        [
            {
                "ligand": "/".join(_u["codes"]),
                "structures": len(_u["pdb_ids"]),
                "poses": _u["poses"],
                "clusters": ", ".join(str(_c) for _c in _u["clusters"]),
                "formula": CalcMolFormula(_mol),
                "mw": round(Descriptors.MolWt(_mol), 1),
                "clogp": round(Descriptors.MolLogP(_mol), 2),
                "hbd": Descriptors.NumHDonors(_mol),
                "hba": Descriptors.NumHAcceptors(_mol),
                "tpsa": round(Descriptors.TPSA(_mol), 1),
                "rot_bonds": Descriptors.NumRotatableBonds(_mol),
                "pdb_ids": ", ".join(_u["pdb_ids"]),
                "smiles": _u["canonical"],
            }
            for _u, _mol in zip(_shown, _mols)
        ]
    )

    _notes = [
        f"{len(_shown)} of {len(_unique)} distinct molecules drawn"
        if len(_shown) < len(_unique)
        else f"{len(_unique)} distinct molecules",
        f"{len(_rows)} crystallised poses across {_rows['pdb_id'].nunique()} structures",
    ]
    if _outside:
        _notes.append(f"{_outside} rows name a PDB entry outside this run, dropped")
    if n_unparsed:
        _notes.append(f"{n_unparsed} rows RDKit could not parse, dropped")

    mo.vstack(
        [
            mo.Html(f'<div style="overflow-x:auto">{_svg}</div>'),
            mo.md("_" + " · ".join(_notes) + "_"),
        ]
        + ([mo.ui.table(_props, selection=None, pagination=True)] if show_ligand_props.value else [])
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
def _(active_info, dist_df, dist_order, mo, np, pd, px):
    mo.stop(dist_df is None, mo.md("No distance matrix."))

    # Symmetrise once; both the ordering and the plot use this copy.
    _ids = list(dist_df.index)
    _raw = dist_df.loc[_ids, _ids].to_numpy(dtype=float)
    _sym = 0.5 * (_raw + _raw.T)
    np.fill_diagonal(_sym, 0.0)
    _M = pd.DataFrame(_sym, index=_ids, columns=_ids)

    def _leaf_order(ids):
        """Average-linkage leaf order of a subset — puts similar pockets adjacent."""
        if len(ids) < 3:
            return list(ids)
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        _sub = _M.loc[ids, ids].to_numpy(dtype=float)
        _Z = linkage(squareform(_sub, checks=False), method="average")
        return [ids[i] for i in leaves_list(_Z)]

    # ---- Build the row/column order -----------------------------------------
    _cluster_of = dict(zip(active_info["pdb_id"].astype(str), active_info["cluster_id"]))
    _blocks = []  # (cluster label, ids) — only used for the block annotations

    if dist_order.value == "as loaded":
        _order = list(_ids)
    elif dist_order.value == "hierarchical":
        _order = _leaf_order(list(_ids))
    else:  # cluster blocks: cluster-major, hierarchical inside each cluster
        _order = []
        for _c in sorted({_cluster_of.get(str(i), -1) for i in _ids}):
            _members = [i for i in _ids if _cluster_of.get(str(i), -1) == _c]
            _members = _leaf_order(_members)
            _blocks.append((_c, _members))
            _order.extend(_members)

    _mat = _M.loc[_order, _order]

    _fig = px.imshow(
        _mat,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(color="spectral dist"),
        height=620,
    )
    _fig.update_layout(
        title=f"Pairwise spectral distance between pockets · ordered by {dist_order.value}",
        xaxis=dict(tickmode="array", tickvals=list(range(len(_order))), ticktext=_order, tickangle=-90),
        yaxis=dict(tickmode="array", tickvals=list(range(len(_order))), ticktext=_order),
        margin=dict(t=70),
    )

    # ---- Outline the blocks so they read at a glance ------------------------
    _cum = 0
    for _c, _members in _blocks:
        _lo, _hi = _cum - 0.5, _cum + len(_members) - 0.5
        _fig.add_shape(
            type="rect", x0=_lo, x1=_hi, y0=_lo, y1=_hi,
            line=dict(color="#1a1a19", width=2), fillcolor="rgba(0,0,0,0)", layer="above",
        )
        _fig.add_annotation(
            x=(_lo + _hi) / 2, y=1.0, xref="x", yref="paper",
            text=f"cluster {_c}", showarrow=False, yanchor="bottom",
            font=dict(size=12, color="#52514e"),
        )
        _cum += len(_members)

    _fig
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


if __name__ == "__main__":
    app.run()
