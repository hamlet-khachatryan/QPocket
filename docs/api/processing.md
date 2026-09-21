# Processing

## Extraction

::: qpocket.pipelines.pockets
    options:
      members:
        - extract_reference_pocket_bbox
        - extract_pocket_from_structure
        - get_plip_interactions
        - run_analysis_step

## Alignment

::: qpocket.pipelines.align
    options:
      members:
        - process_structure
        - process_structure_interface

## Structures and sequences (pure)

::: qpocket.core.structure

::: qpocket.core.sequence

## Structure files and network

::: qpocket.io.structures

::: qpocket.io.rcsb

## Cleaning

::: qpocket.pipelines.clean
    options:
      members:
        - clean_structure