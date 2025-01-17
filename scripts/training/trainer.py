from pytorch_lightning.cli import LightningCLI

from src.srl.data.srl_data_module import SrlDataModule
from src.srl.model.srl_parser import SrlParser


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.padding_label_id", "model.padding_label_id", apply_on="instantiate"
        )
        parser.link_arguments("data.language_model_name", "model.language_model_name")
        parser.link_arguments("data.vocabulary_path", "model.vocabulary_path")
        parser.link_arguments(
            "data.dependency_labels_vocab_path", "model.dependency_labels_vocab_path"
        )


if __name__ == "__main__":
    cli = CustomCLI(
        SrlParser,
        SrlDataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        seed_everything_default=313,
    )
