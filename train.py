from pathlib import Path

import pytorch_lightning as pl
import torch

from src.utils.params import ParamsBuilder
from src.utils.loaders import create_loaders, load_variable
from src.utils.decorators import timespan
from src.models import get_model
from src.utils.save import create_save_path


@timespan("Training")
def main(config_path: Path, dataset_path: Path, output_path: Path) -> None:
    """
    Main function responsible for training classification model.

    Arguments:
        Path config_path: path to main config (of :class:`DefaultConfig` class)
        Path dataset_path: path to dataset
        Path output_path: path to directory where experiment will be saved

    """

    config = load_variable("config", config_path)
    model = get_model(config)
    manual_seed = config.training.seed

    if manual_seed is not None:
        # logger.info(f"Seed the RNG for all devices with {manual_seed}")
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_loader, test_loader, val_loader = create_loaders(
        path_to_data=dataset_path,
        loading_func=config.training.loader_func,
        bs=config.training.batch_size,
    )

    model_path = create_save_path(".data/models", config.model.name)
    config.save_configs(model_path)

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        default_root_dir=model_path,
        checkpoint_callback=config.training.checkpoint_callback,
        weights_summary="full",
        gpus=-1,
        profiler=True,
        callbacks=config.callbacks.value_list(),
    )
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    parser = ParamsBuilder("Single-particle tracking data clssification train")
    parser.add_config_argument()
    parser.add_dataset_argument()
    parser.add_output_argument()

    args, _ = parser.parse_known_args()

    main(**vars(args))
