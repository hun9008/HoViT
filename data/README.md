# Data Folder

This folder contains scripts and instructions for downloading and preparing the datasets used for training and validation.

## Datasets

We use two datasets from the NCT-CRC-HE series available on Zenodo:

- **Training Set**: [NCT-CRC-HE-100K](https://zenodo.org/record/1214456)
- **Validation Set**: [CRC-VAL-HE-7K](https://zenodo.org/record/1214456)

These datasets include histological images for colorectal cancer detection tasks.

## Instructions

Run the provided bash script to automatically download and extract the datasets:

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

After running the script, the data will be organized into the following structure:

```
data/
├── train/
│   └── [training images]
└── val/
    └── [validation images]
```

## Dataset Reference

If you use these datasets in your research, please cite:

```
Kather, Jakob Nikolas; Halama, Niels; Marx, Alexander (2018):
100,000 histological images of human colorectal cancer and healthy tissue.
Zenodo. https://doi.org/10.5281/zenodo.1214456
```

