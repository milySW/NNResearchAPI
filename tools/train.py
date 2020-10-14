from pathlib import Path

from src import models
from src.loarders import DataLoader
from src.trainer import trainer
from src.utils.configurations import setup
from src.utils.decorators import timespan
from src.utils.loaders import load_variable
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder

logger = get_logger("Trainer")


@timespan("Training")
def main(config_path: Path, dataset_path: Path) -> Path:
    """
    Main function responsible for training classification model.

    Arguments:
        Path config_path: Path to main config (of :class:`DefaultConfig` class)
        Path dataset_path: Path to dataset
    """

    config = load_variable("config", config_path)
    model = models.get_model(config)

    train = config.training
    setup(train_config=train, logger=logger)

    train_loader, val_loader, test_loader = DataLoader.get_loaders(
        path_to_data=dataset_path, config=config
    )

    model.set_example(train_loader, dtype=train.dtype)
    learner = trainer.Trainer(config=config)
    learner.fit(model, train_loader, val_loader)

    if train.test and train.save:
        learner.test(test_dataloaders=test_loader)

    return Path(learner.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = ParamsBuilder("Single-particle tracking data clssification train")
    parser.add_config_argument()
    parser.add_dataset_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
