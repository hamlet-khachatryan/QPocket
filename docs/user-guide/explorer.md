# The explorer

```bash
qpocket explore RUNS          # a folder of runs, or one run
```

A marimo app in the browser with three tabs: **Site**, **Exploration** and
**Screening**. The tab bar stays at the top while you scroll. Switching tabs
recomputes nothing and every control keeps its value. Pick the run at the top.

## Site

What the run found, and changing it.

- **Binding sites**: the site table (structures, holo, apo, CavFind misses, ligands,
  lining residues) and the site picker. Everything on the other tabs follows the
  site picked here.
- **3D view**: the frame protein, the picked site's ligands, and every site's centre
  as a sphere — the picked one red, the others grey; hover a sphere for its name and
  counts. **Ligands shown** caps how many ligands are drawn.
- **Regroup sites**: rebuild every site with other grouping parameters (cavity
  overlap, ligand fraction, box padding, minimum cavity points, minimum structure
  share).
- **Edit sites**: **Merge…** two or more sites, **Drop…** one (its ligands become
  misses with reason `site_dropped`), or **Split…** one into its strict cavity groups.
- **Overrides**: every edit, with whether it applied after the last grouping;
  **Undo** removes one.

Every change shows first what it would do — how many sites before and after, and
which existing sites would lose results (clusterings, finals, screening) — and runs
only on **Confirm**. Edits are stored in the run's config (`sites.overrides`) and
re-applied after every regroup. See [Pocket extraction](pocket-extraction.md).

## Exploration

One site, from clouds to representatives.

- **Steps**: **Shape clustering** (no features needed), **Featurise** (with a
  workers setting that defaults to the run's) and **Two-stage clustering**; each
  shows whether it is done, and the terminal commands for the same.
- **Clustering**: method and percentile cuts with a live preview; **Apply to this run**
  saves the clustering (the previous one goes to `metadata/clustering_history/`).
- **Representatives**: each cluster's medoid by default; pick a cluster and a member
  to swap it. Swaps are recorded, and swapping back to the medoid removes the
  override.
- **Structures**, **Pocket landscape** (one point per pocket; PCA/MDS, UMAP or t-SNE)
  and **Pockets side by side** (3D clouds, coloured by a feature, consensus or
  difference from the representative).
- **Interaction fingerprint**: per residue, how often the site's ligands make each
  kind of contact (left) and how often it lines the pocket (right). By default only
  residues a ligand contacts are shown; **Residues** adds those that only line the
  pocket, or shows only those. Selected residues are highlighted in the 3D viewer.
- **Residue contacts by cluster**: clusters × residues, coloured by how often each
  cluster's ligands contact the residue.
- **Ligands**: one depiction per distinct molecule, with properties.
- **Finalise**: freezes the clustering, representatives, residue profile and ligands
  into `sites/<site>/final/`; earlier finals move to `final/history/`. The box says
  whether the final is current or stale; Screening needs a current one.

## Screening

Cross-docking from the site's final: box, docking settings, run (here, SLURM or a VM
over SSH), analysis and poses. See [Screening: cross-docking](virtual-screening.md).
