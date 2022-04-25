import json
import os
from pathlib import Path

import torch
from knockknock import telegram_sender
from pytorch_lightning.callbacks import (EarlyStopping, GPUStatsMonitor,
                                         LearningRateMonitor,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef


def get_training_callbacks(monitor_metric="val_f1score", min_delta=0.003, patience=3):
    """
    get_training_callbacks - returns a list of callbacks for use with PyTorch Lightning-Flash trainer.

    Args:
        monitor_metric (str, optional): metric to monitor for early stopping
        min_delta (float, optional): minimum change in the monitored quantity to qualify as an improvement,
        patience (int, optional): number of epochs with no improvement after which training will be stopped.

    Returns:
        list: list of callbacks for use with PyTorch Lightning-Flash trainer
    """
    _callbacks = [
        StochasticWeightAveraging(),
        LearningRateMonitor(),
        EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            min_delta=min_delta,
            patience=patience,
        ),
    ]
    if torch.cuda.is_available():
        _callbacks.append(GPUStatsMonitor())
    return _callbacks


def get_pubmed_filenames(dataset: str or Path, data_dir: str or Path, verbose=False):
    data_dir = Path(data_dir)
    if dataset == "pubmed_20k":
        datafile_mapping = {
            "train": data_dir / "train20.csv",
            "val": data_dir / "dev20.csv",
            "test": data_dir / "test20.csv",
        }
    else:
        datafile_mapping = {
            "train": data_dir / "train.csv",
            "val": data_dir / "dev.csv",
            "test": data_dir / "test.csv",
        }

    if verbose:
        print(f"datafile_mapping: {datafile_mapping}")
    return datafile_mapping


def load_metrics_train(num_classes: int, verbose=False):
    """
    load_metrics_train - returns a dict of metrics for use with PyTorch Lightning-Flash trainer.

    Args:
        num_classes (int): number of classes in the dataset
        verbose (bool, optional): verbose output

    Returns:
        dict: dict of metrics for use with PyTorch Lightning-Flash trainer (torchmetrics)
    """

    acc = Accuracy(
        num_classes=num_classes, average="weighted" if num_classes > 2 else "macro"
    )
    f1 = F1Score(
        num_classes=num_classes, average="weighted" if num_classes > 2 else "macro"
    )
    mcc = MatthewsCorrCoef(
        num_classes=num_classes,
    )
    _metrics = [acc, mcc, f1]

    if verbose:
        print(f"metrics: {_metrics}")

    return _metrics


def get_tb_logger(
    log_dir: str or Path,
    dataset: str,
    TRAIN_STRATEGY: str,
    model_tag: str,
    verbose=False,
):
    """
    get_logger - returns a TensorBoardLogger object for use with PyTorch Lightning-Flash trainer.

    Args:
        log_dir (str or Path): required, directory to save logs
        dataset (str): required, dataset name
        TRAIN_STRATEGY (str): required, train strategy
        model_tag (str): required, model tag on HuggingFace models

    Returns:
        TensorBoardLogger: object for logging to TensorBoard
    """
    log_dir = Path(log_dir)
    _session_log = log_dir / f"logs_{dataset}_{TRAIN_STRATEGY}"
    _session_log.mkdir(exist_ok=True)

    MODEL_BACKBONE = model_tag.split("/")[-1]
    MODEL_BACKBONE = MODEL_BACKBONE[:30]  # max 30 chars
    logger = TensorBoardLogger(
        save_dir=_session_log, name=f"txtcls_{dataset}_{MODEL_BACKBONE}"
    )
    if verbose:
        print(f"logs will be saved to: {_session_log}")
    return logger


def get_LR_scheduler_config(
    metric_name: str = "val_f1score", interval="epoch", verbose=False
):
    lr_scheduler_config = {
        # REQUIRED: The scheduler instance
        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        "interval": interval,
        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": metric_name,
        "patience": 1,
        "min_lr": 1e-8,
        "reduce_on_plateau": True,
        # If set to `True`, will enforce that the value specified 'monitor'
        # is available when the scheduler is updated, thus stopping
        # training if not found. If set to `False`, it will only produce a warning
        "strict": True,
        # If using the `LearningRateMonitor` callback to monitor the
        # learning rate progress, this keyword can be used to specify
        # a custom logged name
        "name": None,
    }
    if verbose:
        print(f"lr_scheduler_config: {lr_scheduler_config}")
    return lr_scheduler_config


def get_knockknock_notifier(
    trainer,
    datamodule,
    model,
    train_strategy: str = "freeze",
    UNFREEZE_EPOCH: int = 1,
    api_key: str = "KNOCK_TELEGRAM_API",
    chat_id: str = "KNOCK_TELEGRAM_CHAT",
):
    """
    get_knockknock_notifier - returns a knockknock.telegram_sender.TelegramSender object for use with PyTorch Lightning-Flash trainer.
                                optional dependencies for this function are: knockknock (pip install knockknock)

    Args:
        trainer (PyTorch Lightning LightningFlashTrainer): required, trainer object
        datamodule (PyTorch Lightning LightningDataModule): required, datamodule object
        model (PyTorch Lightning LightningModule): required, model object
        train_strategy (str, optional): train strategy, either "freeze", "no_freeze", "freeze_unfreeze", or "unfreeze_freeze"
        UNFREEZE_EPOCH (int, optional): number of epochs to wait before unfreezing the model
        api_key (str, optional): telegram api key in your PATH / env, default "KNOCK_TELEGRAM_API"
        chat_id (str, optional): telegram chat id in your PATH / env, default "KNOCK_TELEGRAM_CHAT"

    Returns:
        knockknock.telegram_sender.TelegramSender: object for sending telegram messages
    """
    BOT_API: str = os.environ.get(api_key)
    CHAT_ID: int = os.environ.get(chat_id)

    @telegram_sender(token=BOT_API, chat_id=CHAT_ID)
    def knockknock_test_wrap(verbose=False):

        if train_strategy == "full_train":
            trainer.fit(
                model,
                datamodule=datamodule,
            )
        else:
            trainer.finetune(
                model,
                datamodule=datamodule,
                strategy=("freeze_unfreeze", UNFREEZE_EPOCH)
                if train_strategy == "freeze_unfreeze"
                else train_strategy,
                # 'freeze_unfreeze' is a special case
            )

        eval_metrics = trainer.test(
            verbose=verbose,
            datamodule=datamodule,
        )
        return eval_metrics

    return knockknock_test_wrap


def save_train_metadata(session_dir: Path, output_metrics: dict, session_params: dict):
    """
    save_train_metadata - saves training metadata to a json file

    Args:
        session_dir (Path): required, directory to save logs
        output_metrics (dict): required, dictionary of metrics
        session_params (dict): required, dictionary of session parameters
    """

    metrics_out_path = session_dir / "training_metrics.json"
    with open(metrics_out_path, "w") as fp:
        json.dump(output_metrics, fp)

    params_out_path = session_dir / "training_parameters.json"
    with open(params_out_path, "w") as fp:
        json.dump(session_params, fp)

    print(f"saved training metrics to: {metrics_out_path}")
