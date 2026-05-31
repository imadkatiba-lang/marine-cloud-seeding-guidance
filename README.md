# Auditable Marine Cloud Seeding Guidance

This repository provides the source code associated with the manuscript:

**Auditable Marine Cloud Seeding Guidance via Physically Gated Reinforcement Decisions and Three-Mode Mapping**

## Description

The code implements a marine cloud-seeding decision pipeline over the Moroccan Atlantic offshore domain. It combines precipitation prediction, reinforcement-based decision learning, physical gating, and technique-level interpretation.

The workflow includes:

* ERA5 data loading and ocean-mask construction
* baseline-anchored precipitation prediction with DeltaNet
* reinforcement decision learning through the MS-DTAC-6A actor-critic module
* physically gated mapping toward hygroscopic, glaciogenic, dynamic, or NO-GO recommendations
* technique-level interpretation using delta-P linkage when available

## Main script

```bash
python marine_cloud_seeding_pipeline.py
```

## Data

The repository does not include raw ERA5 files or large intermediate arrays. These files are not redistributed because of size and access constraints.

ERA5 data can be obtained from the Copernicus Climate Data Store. The required input files and paths are described in the configuration section of the Python script.

## Outputs

The pipeline generates prediction metrics, reinforcement-learning diagnostics, action maps, technique maps, cell-level recommendation tables, and interpretability summaries.

## Archive

A fixed release of this repository will be archived on Zenodo and cited in the manuscript after DOI generation.

## License

This repository is distributed under the MIT License.
