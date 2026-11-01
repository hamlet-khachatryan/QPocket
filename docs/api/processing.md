# Processing

## Per-structure stages (align, clean, ligands, protonate, contacts, cavities)

::: qpocket.pipelines.structure

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