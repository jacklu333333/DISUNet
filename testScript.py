import argparse
import json
import os
import socket

import colored as cl
import deepspeed
import pytorch_lightning as pl
import torch
import torch.utils

from utils.callbacks import TimeFilterProgressBar
from utils.datasets import MotionIDModule, odomDataModule
from utils.model import DiffusionModelPL, baseLineTestingPL
from utils.utility import set_seed

set_seed(0)
torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset name",
    )
    parser.add_argument(
        "--base_weight_path",
        "-b",
        type=str,
        help="Path to the base weight",
    )

    config = json.load(open("config.json", "r"))
    if parser.parse_args().dataset is not None:
        config["dataset"] = parser.parse_args().dataset
    if parser.parse_args().base_weight_path is not None:
        config["base_weight_path"] = parser.parse_args().base_weight_path

    if config["base_weight_path"] == "":
        model = baseLineTestingPL(config=config)
    else:
        model = DiffusionModelPL(config=config)

    if isinstance(model, baseLineTestingPL):
        version = "baseline"
    else:
        version = None
    logger = pl.loggers.TensorBoardLogger(
        "logs/",
        name=f'diffusion_{config["dataset"]}',
        version=version,
    )

    if config["base_weight_path"] != "":
        # check is it a file or not
        if not os.path.isfile(config["base_weight_path"]):
            print(cl.Fore.yellow + "Using deepspeed to load the model" + cl.Style.reset)
            state_dict = (
                deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
                    config["base_weight_path"]
                )
            )
        else:
            state_dict = torch.load(
                config["base_weight_path"],
                weights_only=False,
            )["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    dm = odomDataModule(
        config=config,
        data_dir="./datasets/" + config["dataset"],
    )

    # check dm is MotionIDModule or not
    if isinstance(dm, MotionIDModule):
        config["dataset"] = "MotionID"

    trainer = pl.Trainer(
        devices=1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="loss/val", patience=3, mode="min", min_delta=0.001
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="loss/val",
                save_top_k=3,
                mode="min",
                filename="model-val_loss{loss/val:05.8f}-epoch{epoch:02d}",
                auto_insert_metric_name=False,
            ),
            TimeFilterProgressBar(
                keep_keywords=["v_num", "loss"],
                remove_keywords=["train_epoch", "total_loss/val"],
            ),
        ],
        limit_test_batches=config["limit_test_batches"],
        logger=logger,
        sync_batchnorm=True,
        gradient_clip_val=config["gradient_clip_val"],
        gradient_clip_algorithm=config["gradient_clip_algorithm"],
        detect_anomaly=True,
    )

    if "base_weight_path" in config and config["base_weight_path"] != "":
        trainer.test(model, dm)
    else:
        print(
            cl.Fore.red,
            "!!! Using the random weight for Testing !!!",
            cl.Style.reset,
        )
        trainer.test(model, dm)
    print(cl.Fore.green + "Complete !" + cl.Style.reset)
