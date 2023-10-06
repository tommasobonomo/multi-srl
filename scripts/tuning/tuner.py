import importlib
from pathlib import Path
from typing import Any, List

import optuna
from omegaconf import OmegaConf
from optuna import Trial
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from tap import Tap
from wasabi import msg

from srl.data.srl_data_module import SrlDataModule
from srl.model.srl_parser import SrlParser


class ScriptArgs(Tap):
    config_paths: List[Path]  # List of paths to the config files to use


def single_train(trial: Trial, datamodule: SrlDataModule, conf: dict) -> float:
    if "accumulate_grad_batches" in conf["trainer"]:
        del conf["trainer"]["accumulate_grad_batches"]
    trainer_conf = conf["trainer"].copy()
    # Trainer callbacks and logger must be instantiated
    trainer_conf["callbacks"] = [
        instantiate_object(callback) for callback in trainer_conf["callbacks"]
    ] + [PyTorchLightningPruningCallback(trial, monitor="val/overall_f1")]
    trainer_conf["logger"] = instantiate_object(trainer_conf["logger"])

    trainer = Trainer(**trainer_conf)

    model_conf = conf["model"].copy()
    model_conf["gnn_hidden_dim"] = trial.suggest_int("gnn_hidden_dim", 100, 1000)
    model_conf["edge_embedding_dim"] = trial.suggest_int("edge_embedding_dim", 10, 500)
    model_conf["num_gnn_layers"] = trial.suggest_int("num_gnn_layers", 1, 5)
    model_conf["num_gnn_heads"] = trial.suggest_int("num_gnn_heads", 1, 5)
    model_conf["gnn_dropout"] = trial.suggest_float("gnn_dropout", 0.0, 0.5)

    model = SrlParser(**model_conf)

    try:
        trainer.fit(model, datamodule=datamodule)
    except ValueError:
        # The model has encountered a NaN loss, so we skip this trial
        msg.fail("NaN loss encountered, skipping trial")
        return 0.0

    return trainer.callback_metrics["val/overall_f1"].item()


def instantiate_object(conf: dict) -> Any:
    class_path = conf["class_path"]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**conf.get("init_args", {}))


def run(args: ScriptArgs) -> None:
    all_confs = [OmegaConf.load(path) for path in args.config_paths]
    conf = OmegaConf.to_container(OmegaConf.merge(*all_confs), resolve=True)

    # Link arguments that should be the same
    conf["model"]["padding_label_id"] = conf["data"]["padding_label_id"]
    conf["model"]["language_model_name"] = conf["data"]["language_model_name"]
    conf["model"]["vocabulary_path"] = conf["data"]["vocabulary_path"]
    conf["model"]["dependency_labels_vocab_path"] = conf["data"].get(
        "dependency_labels_vocab_path", None
    )

    # Instantiate and setup datamodule
    datamodule = SrlDataModule(**conf["data"])
    datamodule.setup("fit")
    datamodule.setup("validate")

    # Trial fn
    trian_fn = lambda trial: single_train(trial, datamodule, conf)

    # Run optuna
    study = optuna.create_study(
        study_name="tuning-gnn",
        storage="sqlite:///tuning-gnn.db",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(trian_fn, n_trials=50)


if __name__ == "__main__":
    args = ScriptArgs().parse_args()
    run(args)
