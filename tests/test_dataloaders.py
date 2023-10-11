from pathlib import Path

import pytest

from src.srl.data.srl_data_module import SrlDataModule

root_path = Path("data/preprocessed/conll2009/")
languages = ["ca", "cs", "de", "en", "es", "zh"]
dep_schemes = ["_conll08", "_ud25", ""]
parameters = [
    (language, dep_scheme) for language in languages for dep_scheme in dep_schemes
]


def iterate_over_dataloader(dataloader):
    for batch in dataloader:
        pass

    return True


@pytest.mark.parametrize("language, dep_scheme", parameters)
def test_datamodule(language, dep_scheme):
    vocabulary_file = "vocabulary.json"

    if dep_scheme == "_conll08":
        dependency_labels_vocab_path = (
            f"resources/conll_{language}_dependency_vocab.json"
        )
    elif dep_scheme == "_ud25":
        dependency_labels_vocab_path = "resources/universal_dependency_vocab.json"
    else:
        dependency_labels_vocab_path = None

    datamodule = SrlDataModule(
        vocabulary_path=str(root_path / language / vocabulary_file),
        train_path=str(root_path / (language + dep_scheme) / f"CoNLL2009_train.json"),
        dev_path=str(root_path / (language + dep_scheme) / f"CoNLL2009_dev.json"),
        test_path=str(root_path / (language + dep_scheme) / f"CoNLL2009_test.json"),
        dependency_labels_vocab_path=dependency_labels_vocab_path,
        language_model_name="xlm-roberta-base",
        batch_size=32,
        num_workers=8,
    )

    datamodule.setup("fit")
    datamodule.setup("validate")
    datamodule.setup("test")

    assert iterate_over_dataloader(datamodule.train_dataloader())
    assert iterate_over_dataloader(datamodule.val_dataloader())
    assert iterate_over_dataloader(datamodule.test_dataloader())
