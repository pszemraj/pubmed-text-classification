"""
    extract_7z_archive.py - a script to extract 7z files from a directory (helper script for make_dataset.py)

    details on the 7z format: https://www.7-zip.org/sdk.html and https://py7zr.readthedocs.io/en/latest/user_guide.html
"""

import argparse
import logging
from pathlib import Path
import sys
import py7zr
_src = Path(__file__).parent.parent
_root = _src.parent
_logs_dir = _root / "logs"
_logs_dir.mkdir(exist_ok=True)
sys.path.append(str(_root.resolve()))
from src.utils import collapse_directory
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
        level=logging.INFO, format=log_fmt,filename=_logs_dir / "extract_7z.log")
def get_parser():
    """
    get_parser - a helper function for the argparse module
    """
    parser = argparse.ArgumentParser(
        description="Extract 7z files from a directory"
    )
    parser.add_argument(
        "-i",
        "--input-path",
        required=True,
        type=str,
        help="path to the 7z archive",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=False,
        type=str,
        default=None,
        help="path to the directory to extract the 7z files to. Defaults to the same directory as the input path in a subdirectory called 'extracted'",
    )
    parser.add_argument(
        '-c',
        '--collapse',
        required=False,
        default=False,
        action='store_true',
        help='if passed, will collapse the extracted files into a single directory (i.e. no subdirectories)',
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print more info",
    )
    return parser

if __name__ == "__main__":

    logging.info("Extracting 7z archive")

    args = get_parser().parse_args()
    input_path = Path(args.input_path)
    assert input_path.exists() and input_path.suffix == ".7z", f"input path must be a 7z file, got {input_path}"
    output_path = Path(args.output_path) if args.output_path else input_path.parent / "extracted"
    output_path.mkdir(exist_ok=True)
    verbose = args.verbose
    collapse = args.collapse
    logging.info(f"input args: {args}")

    with py7zr.SevenZipFile(input_path, 'r') as archive:
        archive.extractall(path=output_path)
    if collapse:
        collapse_directory(output_path)
    if verbose:
        print(f"extracted files to {output_path}")
    logging.info(f"extracted files to {output_path}")