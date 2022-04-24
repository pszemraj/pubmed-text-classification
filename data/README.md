# data

This is the data folder. There are three subfolders:

1. raw
2. interim
3. processed

raw data files are in the `raw` folder (uncleaned, new, etc). The raw data files are processed by running `src/data/make_dataset.py` from the root of the project.

The raw data files are then moved to the interim folder as CSV files as a result of the processing. _Files in interim are **NOT** lower-cased_ but are cleaned. Files in interim are then moved to the processed folder as CSV files once lowercased.

---
