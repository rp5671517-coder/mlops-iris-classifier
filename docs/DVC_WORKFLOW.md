# DVC Workflow

## Local DVC Remote

The project uses a local DVC remote for storing dataset versions.

Remote path:

~/dvc-remote-storage

## DVC Workflow

The basic workflow is:

1. Add the dataset using DVC:
   dvc add data/raw/iris_v1.csv

2. Add the DVC metadata to Git:
   git add data/raw/iris_v1.csv.dvc

3. Commit the metadata:
   git commit -m "data: update dataset version"

4. Push dataset objects to the DVC remote:
   dvc push

## Comparing Dataset Versions

Use the following command to compare the current dataset with an earlier Git version:

dvc diff <version1-commit>

## Restoring Dataset Versions

To restore a historical dataset version:

1. Checkout the required .dvc file from Git.
2. Run:
   dvc checkout

This allows Git to select the dataset version while DVC restores the corresponding actual dataset.