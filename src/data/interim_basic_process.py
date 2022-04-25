"""
    interim_basic_process.py - a super barebones script to process the interim data into processed data. The majority of cleaning work has been done in the src\data\make_dataset.py script.

    Here, two things are done: 1) drop null rows, 2) lowercase the text. The output is saved to the processed directory and are used to train transformer models that were pretrained on lowercased text.
"""

import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

_src = Path(__file__).parent.parent
_root = _src.parent
_logs_dir = _root / "logs"
_logs_dir.mkdir(exist_ok=True)
logfile_path = _logs_dir / "basic_data_processor.log"
logging.basicConfig(
    filename=logfile_path,
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def lower_text(text: str):
    return text.lower()


if __name__ == "__main__":

    logging.info("starting running interim -> processed basic")
    _src = _root / "data" / "interim"
    _out = _root / "data" / "processed" / "basic_processed"
    _out.mkdir(exist_ok=True)
    target_col = "description_cln"  # the column to be processed
    files = [f for f in _src.iterdir() if f.is_file() and f.suffix == ".csv"]
    print(len(files))

    for f in tqdm(files, total=len(files)):
        _df = pd.read_csv(f).convert_dtypes()
        _df.dropna(inplace=True)
        _df[target_col] = _df[target_col].apply(lower_text)

        _df.to_csv(_out / f.name, index=False)

    logging.info("finished running interim -> processed basic")
    print("finished running interim -> processed basic")
