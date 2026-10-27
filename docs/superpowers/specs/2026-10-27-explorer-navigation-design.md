# Explorer navigation for large sites, and percentile cuts

Date: 2026-10-27
Status: approved 2026-10-27

## Why

Cavity-defined sites put hundreds of structures in one site (EV-A71 2A:
`S1-G127` holds 828). The explorer was built for tens:

- the shape dendrogram and the mirrored two-stage tree draw every leaf, so at
  828 they are an unreadable wall;
- the landscape prints every PDB ID next to its point;
- the distance heatmap is 828 × 828;
- the point-cloud grid opens every pocket of a cluster at once;
- cut heights are raw, unitless distances (shape ≈ 0.035 at most, chemistry
  ≈ 400), so a cut value means nothing to a reader and does not transfer
  between metrics or sites.

## Decisions (the user's)

0. **Same setup as before, visually different.** The explorer keeps its
   sections, controls and data flow (preview with sliders, "Apply to this
   run", methods, cloud-viewer modes); this work changes how things are
   drawn and how a cut is expressed, not what the page does.
   **The cut slider slides the threshold**: it sets the cut percentile and the
   preview (cut line, leaf colours, cluster counts, table) follows live.

1. **Percentile is the only cut criterion.** A cut is a percentile `p`; the
   tree is still built from the real distances and is cut at the distance
   value of the `p`-th percentile of the site's pairwise distances.
2. **Two-stage chemistry percentile is measured on the whole site** (one
   chemistry scale): "p20" means the same chemical spread in every shape
   cluster, and a chemically uniform shape cluster is not split.
3. **Tree navigation is a collapsed dendrogram plus a cluster table**
   (mockup option A): the tree is drawn only down to the cut, one leaf per
   cluster labelled with its size; next to it a table per cluster. Clicking a
   leaf or a row selects that cluster for the whole page and expands its own
   subtree.
4. **With physicochemical distances, the mirrored tree** keeps the shape tree
   on top (collapsed at the shape cut) and hangs each shape cluster's
   chemistry subtree below it, both axes in percentiles, one chemistry cut
   line across all subtrees.
5. **Landscape: no printed labels**, PDB ID (and cluster, holo/apo) on hover.
6. **Point clouds: 40 shown by default**, chosen in a searchable multi-select
   dropdown.
7. **Figures in a minimal, Nature-paper style** with colour: one hue per shape
   cluster, tints of that hue for its chemistry subclusters, the same colours
   in the tree, table, landscape and clouds; branches coloured below the cut
   and grey above; thin axes, no gridlines, units in axis labels.
   Approved mockup: `.superpowers/brainstorm/14714-1793060409/content/mirrored-cut-v2.html`.

## How (my calls, marked; the builder may refine)

- **Percentile definition** (*my call*): empirical quantile of the
  upper-triangle pairwise distances of the site's matrix (every structure with
  a cloud); `p` in (0, 100). Chemistry uses the site's chemistry matrix;
  single mode uses the combined matrix.
- **Axes in percentiles** (*my call*): merge heights are mapped through the
  site's empirical CDF of pairwise distances, so tree axes read 0–100. The
  mapping is monotone, so tree topology and the cut are unchanged; only the
  vertical scale is stretched. Hover on a merge shows raw distance and
  percentile.
- **Defaults** (*my call*): shape `p15`, chemistry `p20`.
- **Scope of "percentile only"** (*my call*): percentile replaces
  `criterion: count | threshold | relative` and `k` / `threshold` for every
  hierarchical cut (two-stage shape, two-stage chemistry, single-mode
  hierarchical, shape-only). `kmedoids` keeps its `k` and `affinity` its own
  parameters, because neither cuts a tree. The config key becomes
  `distance.clustering.<stage>.percentile`; old `criterion` / `threshold` /
  `k` on a hierarchical stage are dropped by the migration machinery with a
  note, and the stage takes the default percentile.
- **Section names** (*my call*; the user asked for something more sensible
  than panel letters): the existing sections in their existing order, with
  short headings and no panel letters or numbers. No new section is added
  (an earlier mockup's "Chemistry" summary is out of scope).
- **Preview vs applied** (the user: "setup as previous"): as today, the
  Clustering section previews live from its sliders and "Apply to this run"
  saves; every view below reads the applied clustering. Colours come from the
  applied cluster ids; the preview tree uses the same palette by preview id
  and says it is a preview until applied.
- **Small clusters** (*my call*, open to the user's veto): clusters smaller
  than `distance.clustering.min_cluster_size` (4) are drawn grey and pooled
  in the collapsed tree and table as one "small clusters (n)" leaf/row,
  expandable; only clusters at or above that size get their own hue. On
  S1-G127 a p15 shape cut gives 36 clusters, 12 of them singletons, so
  without this the palette would cycle several times.
- **Selection** (*my call*): a table-row selection is the guaranteed way to
  pick a cluster; leaf clicks are added only if marimo 0.24.2 reports them
  reliably on marker points. A row selects its own level (shape cluster or
  chemistry subcluster). Between a row selection and a landscape lasso, the
  most recent one wins (today the lasso sets the clouds' starting cluster;
  that stays).
- **Cloud viewer modes** (*my call*): "one cluster" and "every cluster's
  representative" stay; the representative mode becomes a preset of the
  multi-select. Consensus occupancy and difference-from-representative are
  computed over the whole cluster, never only over the pockets on screen; a
  mixed-cluster selection colours each pocket against its own cluster's
  representative.
- **Cluster table columns**: cluster (swatch), n, holo share, shape within,
  chemistry within (two-stage only), shared volume, representative. Two-stage
  rows expand into chemistry subclusters (indented, tinted swatch).
  - *within* (*my call*): median within-cluster pairwise distance, reported
    as a site percentile, with a small bar.
  - *shared volume* (*my call*): median, over member pairs (sampled, at most
    200 pairs per cluster, fixed seed), of the overlap of two pockets' cavity
    points in the common frame — points rounded to the CavFind grid, overlap =
    |A∩B| / |A∪B| — shown as a percentage.
- **Expanded subtree**: the selected cluster's member-level dendrogram, leaf
  labels hidden, ID on hover; for two-stage, the selected shape cluster's
  chemistry subtree.
- **Cluster distances section** (the existing distance-matrix section, redrawn): a k × k map of median between-cluster
  distances in site percentiles (shape / chemistry / combined switch).
  Selecting a cell shows the member-level block for those two clusters (rows
  ordered by cluster, then leaf order). The full n × n heatmap is no longer
  drawn.
- **Landscape**: small (~3 px), semi-transparent points coloured by cluster
  (or by a chosen feature, diverging scale around 0); lasso selection
  unchanged.
- **Pocket clouds**: the multi-select lists every pocket of the site,
  searchable by ID; options labelled `id · cluster · holo|apo`; default
  selection = cluster representatives first, then members of the selected
  cluster in master order, up to 40; a note says how many exist. Cap 40
  enforced: a larger selection shows the first 40 and says so (*my call*).
- **Colour** (*my call*): a fixed colour-blind-safe qualitative palette of
  ≥ 10 hues for shape clusters; tints at fixed lightness steps for chemistry
  subclusters; more clusters than hues cycle with a note. Above-cut branches
  grey `#9aa3ad`; cut lines dashed. One Plotly template (Helvetica/Arial
  10 pt, axis lines only, ticks outside, no gridlines, white background)
  applied to every figure.

## Goals

- At 828 structures every section renders, reads, and responds.
- A cut value means the same thing for shape, chemistry and combined, in
  every site.

## Non-goals

- Changing how distances are computed, or the clustering linkages.
- The ligand multi-site assignment question (open with the user separately).

## Anti-goals

- A view that silently drops structures without saying how many it hid.
- Colours that disagree between views for the same cluster.

## Acceptance

- Unit tests (TDD): percentile → cut height; axis mapping monotone and
  0–100; site-wide chemistry percentile; legacy criterion configs migrate
  with a note; cluster table values (within, shared volume) on a fixture;
  colour assignment stable and consistent per cluster; point-cloud default
  selection and the 40 cap; landscape has no text labels.
- Explorer render tests pass on the fixture run.
- Full suite green.
- Open `~/qpocket_runs/EV71_2A_cavity` and check `S1-G127` visually.
