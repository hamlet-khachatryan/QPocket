# API Reference

QPocket can be used as a library: pure computations live in `qpocket.core`, file and
tool access in `qpocket.io`, the stages over a run folder in `qpocket.pipelines`, and
the user-facing steps (sites, screening, comparison) in `qpocket.arms`.

| Page | Covers |
|---|---|
| [Library use](library.md) | comparing pockets without a run; the pipelines as functions |
| [Pipeline API](pipeline.md) | a whole run, binding sites, per-site steps, Finalise, clustering, cross-docking |
| [Configuration](config.md) | configuration dataclasses, loading and validation |
| [Processing](processing.md) | per-structure stages, alignment, cleaning, structure files |
| [Distance logic](distance.md) | geodesics, spectral fingerprints, featurisation, external tools |
| [Utilities](utils.md) | run layout, final metadata, geometry, interactions |
