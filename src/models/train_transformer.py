# -*- coding: utf-8 -*-
"""
train_transformer.py - train a transformer model on the data to classify the text. Under the hood, this is ultimately using PyTorch through the lightning-flash library.

- docs: https://lightning-flash.readthedocs.io/en/stable/reference/text_classification.html
- base model: https://huggingface.co/bert-base-uncased

Original file is located at
    https://colab.research.google.com/drive/1Pss62i0H4H8I-wq8wUAoYKRvEJXIuf4e
"""

import argparse
import logging
import os
import pprint as pp
import re
import sys
from pathlib import Path

import flash
import pandas as pd
import torch
from flash.text import TextClassificationData, TextClassifier
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef

_src = Path(__file__).parent.parent
_root = _src.parent
_logs_dir = _root / "logs"
_logs_dir.mkdir(exist_ok=True)
sys.path.append(str(_root.resolve()))

logfile_path = _logs_dir / "train_transformer.log"
logging.basicConfig(
    filename=logfile_path,
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)





#@title nn training parameters
NUM_EPOCHS =  12#@param {type:"integer"}
BATCH_SIZE =  32#@param {type:"integer"}
MAX_LEN = 256 #@param ["128", "256", "512", "1024"] {type:"raw"}
TRAIN_FP16 = True #@param {type:"boolean"}
TRAIN_STRATEGY = "no_freeze" #@param ["freeze", "freeze_unfreeze", "no_freeze", "full_train"]
LR_INITIAL =  1e-4#@param {type:"number"}
LR_SCHEDULE = "reducelronplateau" #@param ["constantlr", "reducelronplateau"]
WEIGHT_DECAY = 0.05 #@param ["0", "0.01", "0.05", "0.1"] {type:"raw"}

#@markdown `UNFREEZE_EPOCH` is the epoch to unfreeze all model layers, only used for
#@markdown `TRAIN_STRATEGY = "freeze_unfreeze"`
UNFREEZE_EPOCH =  4#@param {type:"integer"}

#@markdown `prajjwal1/bert-medium` is default
hf_tag = "bert-base-uncased"

if not torch.cuda.is_available():
    print("cuda not available, setting var TRAIN_FP16 to False.")
    TRAIN_FP16=False


if TRAIN_STRATEGY =="freeze_unfreeze":
    assert NUM_EPOCHS > UNFREEZE_EPOCH > 0, "Please configure params such that NUM_EPOCHS > UNFREEZE_EPOCH > 0"

session_params = {
    "NUM_EPOCHS":NUM_EPOCHS,
    "BATCH_SIZE":BATCH_SIZE,
    "MAX_INPUT_LENGTH":MAX_LEN,
    "TRAIN_FP16":TRAIN_FP16,
    "TRAIN_STRATEGY":TRAIN_STRATEGY,
    "LR_SCHEDULE":LR_SCHEDULE,
    "LR_INITIAL":LR_INITIAL,
    "WEIGHT_DECAY":WEIGHT_DECAY,
    "UNFREEZE_EPOCH":UNFREEZE_EPOCH,
    "hf_tag":hf_tag,

}
logging.info(f"\n\nParameters for a new session:\n\t{session_params}")


dataset = "pubmed_full" #@param ["pubmed_full", "pubmed_20k"]

session_params['dataset'] = dataset # log
if dataset == "pubmed_20k":
    datafile_mapping = {
        "train":data_dir / 'train20.csv',
        "val":data_dir / 'dev20.csv',
        "test":data_dir / 'test20.csv',
    }
else:
    datafile_mapping = {
        "train":data_dir / 'train.csv',
        "val":data_dir / 'dev.csv',
        "test":data_dir / 'test.csv',
    }

# NOTE: this ^ may need some manual exceptions
do_lowercase = "uncased" in hf_tag.lower() or "albert" in hf_tag.lower()
logging.info(f"lowercase={do_lowercase}")

session_params["lowercased_input"] = do_lowercase

input_text_colname = "description_cln" #@param {type:"string"}
target_cls_colname = "target" #@param {type:"string"}


datamodule = TextClassificationData.from_csv(
    input_field=input_text_colname,
    target_fields=target_cls_colname,
    train_file=datafile_mapping["train"],
    val_file=datafile_mapping["val"],
    test_file=datafile_mapping["test"],
    batch_size=BATCH_SIZE,
)

session_params['input_text_colname'] = input_text_colname
session_params['target_cls_colname'] = target_cls_colname



_nc = datamodule.num_classes # alias
print(f"found number of classes as {_nc}")
logging.info(f"found number of classes as {_nc}")
acc = Accuracy(num_classes=_nc, average='weighted' if _nc > 2 else 'macro')
f1 = F1Score(num_classes=_nc, average='weighted' if _nc > 2 else 'macro')
mcc = MatthewsCorrCoef(num_classes=_nc, )
_metrics = [acc, mcc, f1]

session_params["num_classes"] = _nc

"""### training logs"""

from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

log_dir = _root / f"logs_{dataset}_{TRAIN_STRATEGY}"
log_dir.mkdir(exist_ok=True)

# logger = CSVLogger(save_dir=str(log_dir.resolve())) #backup
log_dir_str = str(log_dir.resolve())
MODEL_BACKBONE = hf_tag.split('/')[-1] # parse "microsoft/BiomedNLP-PubMedBERT" with no /
MODEL_BACKBONE = MODEL_BACKBONE[:30] # max 30 chars
logger = TensorBoardLogger(
            save_dir=log_dir_str,
            name=f"txtcls_{dataset}_{MODEL_BACKBONE}"
            )

# log important hyperparameters for setup
session_params["model_shortname"] = MODEL_BACKBONE

logger.log_hyperparams(session_params)

"""### load model

- from huggingface

Q: what models can you actually use?

A:
> AutoModelForSequenceClassification.
Model type should be one of YosoConfig, NystromformerConfig, QDQBertConfig, FNetConfig, PerceiverConfig, GPTJConfig, LayoutLMv2Config, PLBartConfig, RemBertConfig, CanineConfig, RoFormerConfig, BigBirdPegasusConfig, GPTNeoConfig, BigBirdConfig, ConvBertConfig, LEDConfig, IBertConfig, MobileBertConfig, DistilBertConfig, AlbertConfig, CamembertConfig, XLMRobertaXLConfig, XLMRobertaConfig, MBartConfig, MegatronBertConfig, MPNetConfig, BartConfig, ReformerConfig, LongformerConfig, RobertaConfig, DebertaV2Config, DebertaConfig, FlaubertConfig, SqueezeBertConfig, BertConfig, OpenAIGPTConfig, GPT2Config, TransfoXLConfig, XLNetConfig, XLMConfig, CTRLConfig, ElectraConfig, FunnelConfig, LayoutLMConfig, TapasConfig, Data2VecTextConfig.

#### Learning Rate Schedule & Optimizer

- [docs](https://lightning-flash.readthedocs.io/en/stable/general/optimization.html) from lightning-flash
"""

lr_scheduler_config = {
    # REQUIRED: The scheduler instance
    # The unit of the scheduler's step size, could also be 'step'.
    # 'epoch' updates the scheduler on epoch end whereas 'step'
    # updates it after a optimizer update.
    "interval": "epoch",
    # How many epochs/steps should pass between calls to
    # `scheduler.step()`. 1 corresponds to updating the learning
    # rate after every epoch/step.
    "frequency": 1,
    # Metric to to monitor for schedulers like `ReduceLROnPlateau`
    "monitor": "val_f1score",
    "patience": 1,
    "min_lr": 1e-8,
    "reduce_on_plateau":True,
    # If set to `True`, will enforce that the value specified 'monitor'
    # is available when the scheduler is updated, thus stopping
    # training if not found. If set to `False`, it will only produce a warning
    "strict": True,
    # If using the `LearningRateMonitor` callback to monitor the
    # learning rate progress, this keyword can be used to specify
    # a custom logged name
    "name": None,
}


logging.info("\n"*3)
logging.info(f"Loading new model: {hf_tag} for training on text classification")
model = TextClassifier(backbone=hf_tag,
                        max_length=MAX_LEN,
                        labels=datamodule.labels,
                        metrics=_metrics,
                        learning_rate=LR_INITIAL,
                        optimizer=("Adam", {"amsgrad": True,
                                            "weight_decay":WEIGHT_DECAY}),
                        # lr_scheduler=LR_SCHEDULE,
                        lr_scheduler=("reducelronplateau",
                                      {"mode": "max"},
                                      lr_scheduler_config,
                                    ),
                    )
model.hparams.batch_size = BATCH_SIZE

model.configure_optimizers()

"""### create trainer

**deepspeed**
- [docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#deepspeed) from PL
- [page](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#advanced-gpu-optimizations) on advanced GPU optimization
"""

import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DeepSpeedPlugin

seed_everything(42, workers=True)

"""dir(pytorch_lightning.callbacks"""

from pytorch_lightning.callbacks import (EarlyStopping, GPUStatsMonitor,
                                         LearningRateMonitor,
                                         StochasticWeightAveraging)

_callbacks = [
                StochasticWeightAveraging(),
                LearningRateMonitor(),
                GPUStatsMonitor(),
                EarlyStopping(monitor='val_f1score',
                              mode='max',
                              min_delta=0.003,
                              patience=2,
                            ),
            ]
trainer = flash.Trainer(
    max_epochs=NUM_EPOCHS,
    gpus=torch.cuda.device_count(),
    auto_lr_find=True,
    auto_scale_batch_size=True,
    precision=16 if TRAIN_FP16 else 32,
    callbacks=_callbacks,
    logger=logger,
)

"""# train"""

# Commented out IPython magic to ensure Python compatibility.
#@title tensorboard
# Load the TensorBoard notebook extension
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir $log_dir

"""## train trainer"""

logging.info(f"\t\tTRAINING:{hf_tag} ")

if TRAIN_STRATEGY == 'full_train':
    trainer.fit(
        model,
        datamodule=datamodule,
    )
else:
    trainer.finetune(
        model,
        datamodule=datamodule,
        strategy=("freeze_unfreeze", UNFREEZE_EPOCH) if TRAIN_STRATEGY =="freeze_unfreeze" \
                    else TRAIN_STRATEGY, # 'freeze_unfreeze' is a special case
        )

"""# evaluation / prediction"""

# uncomment below if removing knockknock_test_wrap()
trainer.test(verbose=True, datamodule=datamodule,)

# 4. Classify a few sentences

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

"""## extract key metrics"""

import pprint as pp

import numpy

final_metrics = trainer.logged_metrics
output_metrics = {k:v.cpu().numpy().tolist() for k, v in final_metrics.items()}
output_metrics["date_run"] = get_timestamp()
output_metrics["huggingface_tag"] = hf_tag

log_str = f"\t\t\t ======= Final results for {hf_tag} ======="
logging.info(log_str.upper())
logging.info(output_metrics)
pp.pprint(output_metrics)

"""# save & export"""

import gc

gc.collect()

out_dir = _root / "model-checkpoints"
out_dir.mkdir(exist_ok=True)
m_name = hf_tag.split('/')[-1]
_chk_dir = f"{dataset}={m_name}_{get_timestamp()}"
session_dir = out_dir / _chk_dir
session_dir.mkdir(exist_ok=True)

_chk_name = f"textclassifer_{m_name}_{dataset}.pt"
model_out_path = session_dir / _chk_name
trainer.save_checkpoint(model_out_path.resolve())

import json

metrics_out_path = session_dir / "training_metrics.json"
with open(metrics_out_path, "w") as fp:
    json.dump(output_metrics, fp)

params_out_path = session_dir / "training_parameters.json"
with open(params_out_path, "w") as fp:
    json.dump(output_metrics, fp)

import shutil

shutil.copy(logfile_name, session_dir / "training_session_toplevel_log.log")

"""# print log"""

# !cat $_das_logfile

