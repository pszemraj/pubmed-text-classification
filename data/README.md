# Data

This is the data folder. There are three subfolders:

1. raw
2. interim
3. processed

raw data files are in the `raw` folder (uncleaned, new, etc). The raw data files are processed by running `src/data/make_dataset.py` from the root of the project.

The raw data files are then moved to the interim folder as CSV files as a result of the processing. _Files in interim are **NOT** lower-cased_ but are cleaned. Files in interim are then moved to the processed folder as CSV files once lowercased.

## Extracting data

The full pubmed dataset is quite large, and therefore we only extract a small subset of the data directly as .txt / .csv files. The full dataset is available as .7z files in the `data/raw` folder, and processed as archived .csv files in the `data/processed` folder.

Extracting the .7z is possible as an argparse argument when running `src/data/make_dataset.py`. To extract the .7z files outside of that use case, `src\data\extract_7z_archive.py` can be used.

---
