# -*- coding: utf-8 -*-
"""
train_transformer.py - train a transformer model on the data to classify the text. Under the hood, this is ultimately using PyTorch through the lightning-flash library.

- docs: https://lightning-flash.readthedocs.io/en/stable/reference/text_classification.html
- base model: https://huggingface.co/bert-base-uncased

Original file is located at
    https://colab.research.google.com/drive/1Pss62i0H4H8I-wq8wUAoYKRvEJXIuf4e
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

import flash
import pandas as pd
import torch
from flash.text import TextClassificationData, TextClassifier
from pytorch_lightning import seed_everything

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


if __name__ == "__main__":

    logging.info("starting new training session")
    seed_everything(42, workers=True)

    args = get_parser().parse_args()

    NUM_EPOCHS = 12  # @param {type:"integer"}
    BATCH_SIZE = 32  # @param {type:"integer"}
    MAX_LEN = 256  # @param ["128", "256", "512", "1024"] {type:"raw"}
    TRAIN_FP16 = True  # @param {type:"boolean"}
    TRAIN_STRATEGY = (
        "no_freeze"  # @param ["freeze", "freeze_unfreeze", "no_freeze", "full_train"]
    )
    LR_INITIAL = 1e-4  # @param {type:"number"}
    LR_SCHEDULE = "reducelronplateau"  # @param ["constantlr", "reducelronplateau"]
    WEIGHT_DECAY = 0.05  # @param ["0", "0.01", "0.05", "0.1"] {type:"raw"}

    UNFREEZE_EPOCH = 4  # @param {type:"integer"}
    VERBOSE = False  # @param {type:"boolean"}
    hf_tag = "bert-base-uncased"  # @param ["bert-base-uncased", "bert-large-uncased", "bert-base-cased", "bert-large-cased"]
    dataset = "pubmed_full"
    input_text_colname = "description_cln"
    target_cls_colname = "target"
    if not torch.cuda.is_available():
        print("cuda not available, setting var TRAIN_FP16 to False.")
        TRAIN_FP16 = False

    if TRAIN_STRATEGY == "freeze_unfreeze":
        assert (
            NUM_EPOCHS > UNFREEZE_EPOCH > 0
        ), "configure params such that NUM_EPOCHS > UNFREEZE_EPOCH > 0"

    session_params = {
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_INPUT_LENGTH": MAX_LEN,
        "TRAIN_FP16": TRAIN_FP16,
        "TRAIN_STRATEGY": TRAIN_STRATEGY,
        "LR_SCHEDULE": LR_SCHEDULE,
        "LR_INITIAL": LR_INITIAL,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "UNFREEZE_EPOCH": UNFREEZE_EPOCH,
        "hf_tag": hf_tag,
    }
    logging.info(f"\n\nParameters for a new session:\n\t{session_params}")

    session_params["dataset"] = dataset  # log

    datafile_mapping = get_pubmed_filenames(dataset, _root)

    session_params["input_text_colname"] = input_text_colname
    session_params["target_cls_colname"] = target_cls_colname

    datamodule = TextClassificationData.from_csv(
        input_field=input_text_colname,
        target_fields=target_cls_colname,
        train_file=datafile_mapping["train"],
        val_file=datafile_mapping["val"],
        test_file=datafile_mapping["test"],
        batch_size=BATCH_SIZE,
    )

    _nc = datamodule.num_classes  # alias
    logging.info(f"found number of classes as {_nc}")
    session_params["num_classes"] = _nc

    _metrics = load_metrics_train(num_classes=_nc, verbose=VERBOSE)

    logger = get_tb_logger(
        log_dir=_logs_dir, dataset=dataset, model_tag=hf_tag, verbose=VERBOSE
    )

    logger.log_hyperparams(session_params)

    lr_scheduler_config = get_LR_scheduler_config()
    logging.info(f"lr_scheduler_config={lr_scheduler_config}")
    logging.info(f"Loading model: {hf_tag} for training on text classification")
    model = TextClassifier(
        backbone=hf_tag,
        max_length=MAX_LEN,
        labels=datamodule.labels,
        metrics=_metrics,
        learning_rate=LR_INITIAL,
        optimizer=("Adam", {"amsgrad": True, "weight_decay": WEIGHT_DECAY}),
        lr_scheduler=(
            "reducelronplateau",
            {"mode": "max"},
            lr_scheduler_config,
        ),
    )
    model.hparams.batch_size = BATCH_SIZE
    model.configure_optimizers()

    trainer = flash.Trainer(
        max_epochs=NUM_EPOCHS,
        gpus=torch.cuda.device_count(),
        auto_lr_find=True,
        auto_scale_batch_size=True,
        precision=16 if TRAIN_FP16 else 32,
        callbacks=get_training_callbacks(
            monitor_metric="val_f1score", min_delta=0.003, patience=2
        ),
        logger=logger,
    )

    logging.info(f"\t\tTRAINING:{hf_tag} ")

    if TRAIN_STRATEGY == "full_train":
        trainer.fit(
            model,
            datamodule=datamodule,
        )
    else:
        trainer.finetune(
            model,
            datamodule=datamodule,
            strategy=("freeze_unfreeze", UNFREEZE_EPOCH)
            if TRAIN_STRATEGY == "freeze_unfreeze"
            else TRAIN_STRATEGY,  # 'freeze_unfreeze' is a special case
        )

    trainer.test(
        verbose=False,
        datamodule=datamodule,
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
    out_dir = _root / "model-checkpoints"
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
