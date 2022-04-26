"""
    extract_tensorboard_logs.py - iterate through tensorboard's wonderful and easy to parse log files
    NOTE: as this script is optional, it requires an additional dependency: tensorflow.
"""
import argparse
import logging

import os
from pathlib import Path
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from tqdm.auto import tqdm
from datetime import datetime

_root_dir = Path(__file__).parent.parent.parent
logging.basicConfig(level=logging.INFO)


def get_timestamp():
    # return current date and time as as string
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def convert_tfevent(filepath, log_examples=False):

    _f = Path(filepath)

    _df = pd.DataFrame(
        [
            parse_tfevent(e, log_examples)
            for e in summary_iterator(filepath)
            if len(e.summary.value)
        ]
    )

    _df["version"] = _f.parent.name
    _df["model"] = _f.parent.parent.name
    _df["dataset"] = str(_f.parent.parent.parent.name).replace("logs_", "")

    return _df


def parse_tfevent(tfevent, log_examples=False):
    event_data = dict(
        wall_time=tfevent.wall_time,
        name=tfevent.summary.value[0].tag,
        step=tfevent.step,
        value=float(tfevent.summary.value[0].simple_value),
    )
    if log_examples:
        logging.info(f"\n\nEvent data for example:\n\t{pp.pformat(event_data)}")
        logging.info(f"other fields:\t{pp.pformat(dir(tfevent))}")

    return event_data


def convert_tb_data(root_dir, sort_by="step"):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """

    columns_order = [
        "dataset",
        "model",
        "version",
        "wall_time",
        "name",
        "step",
        "value",
    ]
    out = []
    for (root, _, filenames) in tqdm(os.walk(root_dir)):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            try:
                file_full_path = os.path.join(root, filename)
                out.append(convert_tfevent(file_full_path, log_examples=False))
            except Exception as e:
                logging.warning(f"unable to convert the event, error is: {e}")

    # Concatenate (and sort) all partial individual dataframes
    if out:
        df = pd.concat(out)
        if sort_by:
            df.sort_values(by=sort_by, inplace=True)
        return df[columns_order]
    else:

        return pd.DataFrame()


def get_parser():
    """
    get_parser - a helper function for the argparse module
    """
    parser = argparse.ArgumentParser(description="Extract   tensorboard logs")
    parser.add_argument(
        "-i",
        "--input-path",
        required=True,
        type=str,
        help="path to the directory containing other directories that are tensorboard logs",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        required=False,
        type=str,
        default=None,
        help="where to save the output file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print more info",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    verbose = args.verbose
    logging.info(f"converting tensorboard logs from {args.input_path}")
    input_path = Path(args.input_path)
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else _root_dir / "reports" / "converted_tensorboard_logs"
    )
    output_path.mkdir(exist_ok=True, parents=True)

    logdirs = [d for d in input_path.iterdir() if d.is_dir() and "logs_" in d.stem]
    logging.info(f"found {len(logdirs)} directories with logs")

    overall_df = pd.DataFrame()
    for logdir in tqdm(logdirs, total=len(logdirs)):
        df = convert_tb_data(logdir).convert_dtypes()
        if len(df) > 0:
            overall_df = pd.concat([overall_df, df], sort=True)
        else:
            logging.warning(f"no data found in {logdir.name}")
    overall_df.reset_index(drop=True, inplace=True)
    pre_len = overall_df.shape[0]
    overall_df.dropna(inplace=True)
    logging.info(f"dropped {pre_len - overall_df.shape[0]} rows with NaN values")
    if verbose:
        logging.info(f"overall_df:\n{overall_df.info()}")
    gpu_stuff = [
        n for n in overall_df.name.unique() if "device_id:" in n or "gpu_id:" in n
    ]
    misc_useless = [
        "hp_metric",
        "_hparams_/experiment",
        "_hparams_/session_start_info",
        "_hparams_/session_end_info",
    ]
    gpu_stuff.extend(misc_useless)
    df_cln = overall_df.drop(overall_df[overall_df.name.isin(gpu_stuff)].index)
    logging.info(f"dropped {overall_df.shape[0] - df_cln.shape[0]} rows with gpu stuff")
    df_cln.to_csv(output_path / f"tensorboard_logs{get_timestamp()}.csv", index=False)
    df_cln.to_excel(
        output_path / f"tensorboard_logs{get_timestamp()}.xlsx", index=False
    )
    logging.info(f"saved to {output_path} with the length of {df_cln.shape[0]} rows")
