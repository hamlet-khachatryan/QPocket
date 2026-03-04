# User Guide
## PDB Structure Fetching Module

The **PDB Query Module** is the entry point for the QPocket pipeline. It is responsible for identifying and downloading protein structures from the RCSB PDB that share sequence identity and biological relevance with a given reference protein.

### Key Features
* Sequence-Based Search: Uses BLAST-like sequence identity thresholds to find homologs.
* Optionally enforces matching UniProt Accession IDs to ensure biological consistency (e.g., distinguishing between very similar isoforms).
* Filters results by experimental method (e.g., X-Ray) and resolution limit.

### Configuration Parameters
These parameters are defined in conf/config.yaml under the search and global sections.

| Parameter                  |Type| Default |Description|
|----------------------------|--------|---------|--------|
| `workdir`                   |`str`| `null`  |Root path for output. If null, defaults to Current Working Directory.|
| `query_reference`            |`str`| `N/A`   |The target PDB and Chain ID (format: PDBID_CHAIN).|
| `search.sequence_identity`   |`float`| 1.0     |Sequence identity cutoff (0.0 to 1.0). 1.0 implies 100% match.|
| `search.resolution_limit`    |`float`| 2.5     |Maximum allowed resolution in Ångstroms. Set to null to disable.|
| `search.experimental_method` |`str`| "NMR"   |Filter by method (e.g., NMR, ELECTRON MICROSCOPY).|
| `search.enforce_same_uniprot` |`bool`| `true`  |If true, only fetches structures that map to the same UniProt ID as the reference.|

### Python API Usage

The module uses two primary functions in `src.pdb_query.api`: `fetch_structure_ids` and `download_structures`.

```python
from src.pdb_query import api
from src import utils

# 1. Setup Project Hierarchy
ref_id = "5R7Y_A"
dirs = utils.setup_directory_hierarchy(workdir=None, ref_pdb_chain=ref_id)

# 2. Fetch Matching IDs
pdb_ids = api.fetch_structure_ids(
    ref_pdb_id="5R7Y",
    ref_chain_id="A",
    similarity_threshold=1.0,    # 100% Identity
    min_resolution=2.5,          # Resolution <= 2.5 Å
    experimental_method="X-RAY DIFFRACTION",
    enforce_uniprot=True         # Must match reference UniProt ID
)

# 3. Download Files
if pdb_ids:
    api.download_structures(pdb_ids, dirs['structures'])
```

