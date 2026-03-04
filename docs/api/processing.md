# Processing

## Extraction

::: qpocket_pipeline.processing.extraction
    options:
      members:
        - extract_reference_pocket_bbox
        - extract_pocket_from_structure
        - get_plip_interactions
        - run_analysis_step

## Alignment

::: qpocket_pipeline.processing.alignment
    options:
      members:
        - process_structure
        - fetch_uniprot_sequence
        - generate_renumbering_map

## Cleaning

::: qpocket_pipeline.processing.cleaning
    options:
      members:
        - clean_structure