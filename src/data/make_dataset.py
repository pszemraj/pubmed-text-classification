# -*- coding: utf-8 -*-
import argparse
import logging
from pathlib import Path
from tqdm.auto import tqdm
from cleantext import clean
import sys
import pandas as pd
import py7zr
import shutil

_src = Path(__file__).parent.parent
_root = _src.parent
_logs_dir = _root / "logs"
_logs_dir.mkdir(exist_ok=True)
sys.path.append(str(_root.resolve()))
from src.utils import collapse_directory, fix_punct_spaces


def process_txt_data(
    txt_datadir: str or Path, out_dir: str or Path, lowercase=True, verbose=False
):
    """read each downloaded txt file into pandas, convert to a dataframe, and save as a CSV"""
    txt_datadir = Path(txt_datadir)
    out_dir = Path(out_dir) if out_dir is not None else txt_datadir / "processed"
    out_dir.mkdir(exist_ok=True)
    # get all txt files in the directory
    text_files = [
        f for f in txt_datadir.iterdir() if f.is_file() and f.suffix == ".txt"
    ]
    csv_paths = []

    for txt_path in tqdm(text_files, total=len(text_files)):

        df = pd.read_csv(
            txt_path,
            skiprows=1,
            delimiter="\t",
            header=None,
            on_bad_lines="skip",
            engine="python",
        ).convert_dtypes()
        df.columns = ["target", "description"]
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["description_cln"] = df["description"].apply(clean, lower=lowercase)
        df["description_cln"] = df["description_cln"].apply(fix_punct_spaces)
        _csv_out_path = out_dir / f"{txt_path.stem}.csv"
        df.to_csv(_csv_out_path, index=False)
        csv_paths.append(_csv_out_path)

    if verbose:
        print(f"processed and returning:\n\t{[f.name for f in csv_paths]}")

    return csv_paths


def main(
    input_path, output_path, lowercase=False, process_zip_file=False, verbose=False
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    input_path = Path(input_path)
    output_path = Path(output_path)
    if process_zip_file:
        zipfiles = [
            f for f in input_path.iterdir() if f.is_file() and f.suffix == ".7z"
        ]
        logger.info(f"extracting zip files.. found {len(zipfiles)} zip files")
        temp_dir = input_path / "temp_7z_dir"
        for f in tqdm(zipfiles, total=len(zipfiles)):
            with py7zr.SevenZipFile(f, "r") as z:
                z.extractall(path=temp_dir)
        collapse_directory(temp_dir)
    # extract standard txt files
    csv_paths = process_txt_data(
        txt_datadir=input_path,
        out_dir=output_path,
        lowercase=lowercase,
        verbose=verbose,
    )
    if process_zip_file:
        _input_dir_count = len(csv_paths)
        logger.info(f"processed {_input_dir_count} files in input_path")
        # extract text files from zip files
        logger.info(f"extracting zip files in {temp_dir.resolve()}")
        csv_paths += process_txt_data(
            txt_datadir=temp_dir,
            out_dir=output_path,
            lowercase=lowercase,
            verbose=verbose,
        )
        zip_count = len(csv_paths) - _input_dir_count
        logger.info(f"processed {zip_count} files in temp_dir")

        # remove temp dir
        logger.info(f"removing temp dir {temp_dir.resolve()}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info(f"returning {len(csv_paths)} csv files")
    if verbose:
        print(
            f"processed and saved:\n\t{[f.name for f in csv_paths]} in directory {output_path}"
        )

    return csv_paths


def get_parser():
    """
    get_parser - a helper function for the argparse module
    """
    parser = argparse.ArgumentParser(
        description="Make a dataset from the raw data",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        required=False,
        type=str,
        default=None,
        help="The path to the input data directory. Defaults to root/data/raw",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=False,
        type=str,
        default=None,
        help="The path to the output data directory. Defaults to root/data/interim",
    )
    parser.add_argument(
        "-z",
        "--process-zip-file",
        required=False,
        default=False,
        action="store_true",
        help="If passed, will also parse the zip files in the input path",
    )
    parser.add_argument(
        "-l",
        "--lowercase",
        required=False,
        default=False,
        action="store_true",
        help="If passed, will lowercase the input text",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        default=False,
        action="store_true",
        help="If passed, will print out the paths to the output files",
    )
    return parser


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_fmt, filename=_logs_dir / "make_dataset.log"
    )

    args = get_parser().parse_args()
    logger = logging.info(f"parsed args: {args}")
    input_dir = Path(args.input_path) if args.input_path else _root / "data" / "raw"
    output_dir = (
        Path(args.output_path) if args.output_path else _root / "data" / "interim"
    )
    assert input_dir.exists(), f"input_dir {input_dir} does not exist"
    assert output_dir.exists(), f"output_dir {output_dir} does not exist"
    process_zip_file = args.process_zip_file
    lowercase = args.lowercase
    verbose = args.verbose
    _ = main(
        input_path=input_dir,
        output_path=output_dir,
        lowercase=lowercase,
        process_zip_file=process_zip_file,
        verbose=verbose,
    )
