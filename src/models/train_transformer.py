# -*- coding: utf-8 -*-
"""
train_transformer.py - train a transformer model on the data to classify the text. Under the hood, this is ultimately using PyTorch through the lightning-flash library.

- docs: https://lightning-flash.readthedocs.io/en/stable/reference/text_classification.html
- base model: https://huggingface.co/bert-base-uncased
- a copy of the original training notebook can be found at https://gist.github.com/pszemraj/746db81efea08cc25305a04f86ec8a65
"""

import argparse
import gc
import json
import logging
import os
import pprint as pp
import re
import shutil
import sys
from pathlib import Path
import warnings

import flash
import pandas as pd
import torch
from flash.text import TextClassificationData, TextClassifier
from pytorch_lightning import seed_everything
from src.models.trf_text import get_knockknock_notifier

from trf_text import (
    get_LR_scheduler_config,
    get_pubmed_filenames,
    get_tb_logger,
    get_training_callbacks,
    load_metrics_train,
    save_train_metadata,
)

_src = Path(__file__).parent.parent
_root = _src.parent
_logs_dir = _root / "logs"
_logs_dir.mkdir(exist_ok=True)
sys.path.append(str(_root.resolve()))

from src.utils import get_timestamp

logfile_path = _logs_dir / "train_transformer.log"
logging.basicConfig(
    filename=logfile_path,
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def predict_test_example():

    test_df = pd.read_csv(datafile_mapping["test"]).convert_dtypes()
    predict_examples = test_df.sample(n=5)
    sample_text = predict_examples[input_text_colname].to_list()
    sample_ytrue = predict_examples[target_cls_colname].to_list()

    datamodule = TextClassificationData.from_lists(
        predict_data=sample_text,
        batch_size=8,
    )
    predictions = trainer.predict(model, datamodule=datamodule, output="labels")[0]

    pp.pprint(dict(zip(sample_text, sample_ytrue)))

    logging.info(f"\nmodel {hf_tag} had the following ytrue:ypred split:\n")
    logging.info(f"\nYtrue:\t{sample_ytrue}\nYpred:\t{predictions} ")
    print(f"\nYtrue:\t{sample_ytrue}\nYpred:\t{predictions}")


def get_parser():
    """
    get_parser - a helper function for the argparse module
    """
    parser = argparse.ArgumentParser(
        description="Train a transformer model on the data to classify the text."
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=12,
        help="Number of epochs to train the model for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training the model.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum length of the input text.",
    )
    parser.add_argument(
        "--train-fp16",
        default=True,
        action="store_true",
        help="Whether to train the model in FP16.",
    )
    parser.add_argument(
        "--train-strategy",
        type=str,
        default="freeze",
        help="Training strategy for the model. Can be any of freeze, freeze_unfreeze, no_freeze, full_train",
    )
    parser.add_argument(
        "--LR-initial",
        type=float,
        default=1e-4,
        help="Initial learning rate for the model.",
    )
    parser.add_argument(
        "--LR-schedule",
        type=str,
        default="reducelronplateau",
        help="Learning rate schedule for the model.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay for the model.",
    )
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=4,
        help="Epoch to unfreeze the model IF using the 'freeze_unfreeze' strategy.",
    )

    parser.add_argument(
        "--hf-tag",
        type=str,
        default="distilbert-base-uncased",
        help="Huggingface model tag (see https://huggingface.co/models). BERT is bert-base-uncased.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pubmed_20k",
        help="Dataset to train the model on. Options are: pubmed_full and pubmed_20k",
    )
    parser.add_argument(
        "-lc",
        "--lowercased-text",
        default=False,
        action="store_true",
        help="Whether to lowercase the text.",
    )
    parser.add_argument(
        "--input-text-colname",
        type=str,
        default="description_cln",
        help="Column name of the input text.",
    )
    parser.add_argument(
        "--target-cls-colname",
        type=str,
        default="target",
        help="Column name of the target class.",
    )
    parser.add_argument(
        "-kk",
        "--knockknock",
        default=False,
        action="store_true",
        help="whether to use knockknock to notify when training completed (note: requires knockknock package and environment vars",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Increase verbosity of logging.",
    )
    return parser


if __name__ == "__main__":

    logging.info("starting new training session")
    seed_everything(42, workers=True)

    args = get_parser().parse_args()
    logging.info(f"args: {pp.pformat(args)}")
    dataset = args.dataset
    lowercased_text = args.lowercased_text
    input_text_colname = args.input_text_colname
    target_cls_colname = args.target_cls_colname
    max_len = args.max_len  # max length of the input text (for BERT / XLNet / others)

    hf_tag = args.hf_tag

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    train_fp16 = args.train_fp16
    train_strategy = args.train_strategy
    LR_initial = args.LR_initial
    LR_schedule = args.LR_schedule
    weight_decay = args.weight_decay
    unfreeze_epoch = args.unfreeze_epoch

    use_knockknock = args.knockknock
    verbose = args.verbose

    if not torch.cuda.is_available():
        warnings.warn("cuda not available, setting var TRAIN_FP16 to False.")
        logging.info("cuda not available, setting var TRAIN_FP16 to False.")
        train_fp16 = False

    if train_strategy == "freeze_unfreeze":
        assert (
            num_epochs > unfreeze_epoch > 0
        ), f"required that NUM_EPOCHS > UNFREEZE_EPOCH > 0, found NUM_EPOCHS={num_epochs} and UNFREEZE_EPOCH={unfreeze_epoch}"

    session_params = {
        "NUM_EPOCHS": num_epochs,
        "BATCH_SIZE": batch_size,
        "MAX_INPUT_LENGTH": max_len,
        "TRAIN_FP16": train_fp16,
        "TRAIN_STRATEGY": train_strategy,
        "LR_SCHEDULE": LR_schedule,
        "LR_INITIAL": LR_initial,
        "WEIGHT_DECAY": weight_decay,
        "UNFREEZE_EPOCH": unfreeze_epoch,
        "hf_tag": hf_tag,
        "dataset": dataset,
        "lowercased_text": lowercased_text,
        "input_text_colname": input_text_colname,
        "target_cls_colname": target_cls_colname,
        "verbose": verbose,
    }
    logging.info(f"\n\nParameters for a new session:\n\t{pp.pformat(session_params)}")
    # load data from appropriate dataset depending on 'lowercased_text'
    src_data_dir = (
        _root / "data" / "processed" if lowercased_text else _root / "data" / "interim"
    )
    datafile_mapping = get_pubmed_filenames(
        data_dir=src_data_dir, dataset=dataset, verbose=verbose
    )

    datamodule = TextClassificationData.from_csv(
        input_field=input_text_colname,
        target_fields=target_cls_colname,
        train_file=datafile_mapping["train"],
        val_file=datafile_mapping["val"],
        test_file=datafile_mapping["test"],
        batch_size=batch_size,
    )

    _nc = datamodule.num_classes  # alias
    logging.info(f"found number of classes as {_nc}")
    session_params["num_classes"] = _nc

    _metrics = load_metrics_train(num_classes=_nc, verbose=verbose)

    logger = get_tb_logger(
        log_dir=_logs_dir,
        dataset=dataset,
        TRAIN_STRATEGY=train_strategy,
        model_tag=hf_tag,
        verbose=verbose,
    )
    logger.log_hyperparams(session_params)

    lr_scheduler_config = get_LR_scheduler_config()
    logging.info(f"lr_scheduler_config={lr_scheduler_config}")
    logging.info(f"Loading model: {hf_tag} for training on text classification")
    model = TextClassifier(
        backbone=hf_tag,
        max_length=max_len,
        labels=datamodule.labels,
        metrics=_metrics,
        learning_rate=LR_initial,
        optimizer=("Adam", {"amsgrad": True, "weight_decay": weight_decay}),
        lr_scheduler=(
            "reducelronplateau",
            {"mode": "max"},
            lr_scheduler_config,
        ),
    )
    model.hparams.batch_size = batch_size
    model.configure_optimizers()

    trainer = flash.Trainer(
        max_epochs=num_epochs,
        gpus=torch.cuda.device_count(),
        auto_lr_find=True,
        auto_scale_batch_size=True,
        precision=16 if train_fp16 else 32,
        callbacks=get_training_callbacks(
            monitor_metric="val_f1score", min_delta=0.003, patience=2
        ),
        logger=logger,
    )

    logging.info(f"starting training of model {hf_tag}")
    if use_knockknock:
        logging.info("using knockknock to notify when training is complete")
        train_with_knockknock = get_knockknock_notifier(
            trainer=trainer,
            datamodule=datamodule,
            model=model,
            train_strategy=train_strategy,
            unfreeze_epoch=unfreeze_epoch,
        )

        eval_results = train_with_knockknock()
    else:
        logging.info("standard PL training")
        if train_strategy == "full_train":
            trainer.fit(
                model,
                datamodule=datamodule,
            )
        else:
            trainer.finetune(
                model,
                datamodule=datamodule,
                strategy=("freeze_unfreeze", unfreeze_epoch)
                if train_strategy == "freeze_unfreeze"
                else train_strategy,  # 'freeze_unfreeze' is a special case
            )

        eval_results = trainer.test(
            verbose=False,
            datamodule=datamodule,
        )
    logging.info(
        f"finished training of model {hf_tag} and evaluation results: {pp.pformat(eval_results)}"
    )
    predict_test_example()  # illustrate how to use predict() and what the results look like

    final_metrics = trainer.logged_metrics
    output_metrics = {k: v.cpu().numpy().tolist() for k, v in final_metrics.items()}
    output_metrics["date_run"] = get_timestamp()
    output_metrics["huggingface_tag"] = hf_tag
    logging.info(output_metrics)
    pp.pprint(output_metrics)

    gc.collect()
    # save model
    out_dir = _root / "models"
    out_dir.mkdir(exist_ok=True)
    m_name = hf_tag.split("/")[-1]
    session_dir = out_dir / f"{dataset}={m_name}_{get_timestamp()}"
    session_dir.mkdir(exist_ok=True)

    model_out_path = session_dir / f"textclassifer_{m_name}_{dataset}.pt"
    trainer.save_checkpoint(model_out_path.resolve())

    save_train_metadata(
        session_dir=session_dir,
        session_params=session_params,
        output_metrics=output_metrics,
    )
    shutil.copyfile(logfile_path, session_dir / "training_transformers.log")

    print("Finished training")
