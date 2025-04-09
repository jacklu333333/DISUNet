import argparse
import json
import os
import socket

import colored as cl
import deepspeed
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DeepSpeedStrategy

from utils.callbacks import TimeFilterProgressBar
from utils.datasets import MotionIDModule, odomDataModule
from utils.model import DiffusionModelPL
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

    logger = pl.loggers.TensorBoardLogger(
        "logs/",
        name=f'diffusion_{config["dataset"]}',
    )

    model = DiffusionModelPL(config=config)
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
        model.load_state_dict(state_dict)
    dm = odomDataModule(
        config=config,
        data_dir="./datasets/" + config["dataset"],
    )
    # check dm is MotionIDModule or not
    if isinstance(dm, MotionIDModule):
        config["dataset"] = "MotionID"

    trainer = pl.Trainer(
        max_epochs=1000,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="loss/val",
                patience=3,
                mode="min",
                min_delta=0.01,
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="loss/val",
                save_top_k=3,
                mode="min",
                filename="model-loss_val{loss/val:05.8f}-step{step:02d}-epoch{epoch:02d}",
                auto_insert_metric_name=False,
            ),
            TimeFilterProgressBar(
                keep_keywords=["v_num", "loss", "noise_mse"],
                remove_keywords=["train_epoch", "loss_"],
            ),
        ],
        logger=logger,
        sync_batchnorm=True,
        gradient_clip_val=config["gradient_clip_val"],
        gradient_clip_algorithm=config["gradient_clip_algorithm"],
        detect_anomaly=True,
    )
    if config["lr_finder"]:
        tuner = pl.tuner.tuning.Tuner(trainer)
        lr_finder = tuner.lr_find(
            model,
            dm,
            min_lr=1e-10,
            max_lr=1e-2,
            mode="exponential",
            update_attr=True,
            attr_name="lr",
        )

    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path="best")

    print(cl.Fore.green + "Complete !" + cl.Style.reset)