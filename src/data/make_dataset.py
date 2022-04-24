# -*- coding: utf-8 -*-
import argparse
import logging
from pathlib import Path
from tqdm.auto import tqdm
from cleantext import clean

import pandas as pd

from src.utils import fix_punct_spaces

_src = Path(__file__).parent.parent
_root = _src.parent

def process_txt_data(txt_datadir:str or Path,
                     out_dir:str or Path=None,
                     lowercase=True,
                     verbose=False):
    """read each downloaded txt file into pandas, convert to a dataframe, and save as a CSV"""
    txt_datadir = Path(txt_datadir)
    out_dir = Path(out_dir) if out_dir is not None else txt_datadir / 'processed'
    out_dir.mkdir(exist_ok=True)
    # get all txt files in the directory
    text_files = [f for f in txt_datadir.iterdir() if f.is_file() and f.suffix == '.txt']
    csv_paths = []

    for txt_path in tqdm(text_files, total=len(text_files)):

        df = pd.read_csv(txt_path,
                         skiprows=1,
                         delimiter='\t',
                         header=None,
                         on_bad_lines='skip',
                         engine='python',
                    ).convert_dtypes()
        df.columns = ['target', 'description']
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["description_cln"] = df["description"].apply(clean, lower=lowercase)
        df["description_cln"] = df["description_cln"].apply(fix_punct_spaces)
        _csv_out_path = out_dir / txt_path.with_suffix('.csv')
        df.to_csv(_csv_out_path, index=False)
        csv_paths.append(_csv_out_path)

    if verbose:
        print(f"processed and returning:\n\t{[f.name for f in csv_paths]}")

    return csv_paths

def main(input_filepath, output_filepath, verbose=False, zip_file=False):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

def get_parser():
    """
    get_parser - a helper function for the argparse module
    """
    parser = argparse.ArgumentParser(
        description='Make a dataset from the raw data')
    )
    parser.add_argument(
        "-i",
        "--input-path",
        required=False,
        type=str,
        default=None,
        help="The path to the input data"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=False,
        type=str,
        default=None,
        help="The path to the output data"
    )
    parser.add_argument(
        '-z',
        '--zip-file',
        required=False,
        default=False,
        action='store_true',
        help='If passed, will also parse the zip files in the input path'
    )
    parser.add_argument(
        '-l',
        '--lowercase',
        required=False,
        default=False,
        action='store_true',
        help='If passed, will lowercase the input text'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        required=False,
        default=False,
        action='store_true',
        help='If passed, will print out the paths to the output files'
    )
    return parser

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]

    args = get_parser().parse_args()
    input_filepath = Path(args.input_path) if args.input_path else project_dir / 'data' /' raw'
    output_filepath = Path(args.output_path) if args.output_path else project_dir / 'data' / 'interim'
    verbose = args.verbose
    main(input_filepath, output_filepath, verbose)
    # not used in this stub but often useful for finding various files


    main()
