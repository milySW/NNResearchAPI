from pathlib import Path

from src.models import get_model
from src.trainer.trainer import Trainer
from src.utils.decorators import timespan
from src.utils.externals_configurations import set_deterministic_environment
from src.utils.loaders import get_loaders, load_variable
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder

logger = get_logger("Trainer")


@timespan("Training")
def main(config_path: Path, dataset_path: Path) -> None:
    """
    Main function responsible for training classification model.

    Arguments:
        Path config_path: path to main config (of :class:`DefaultConfig` class)
        Path dataset_path: path to dataset

    """

    config = load_variable("config", config_path)
    model = get_model(config)
    manual_seed = config.training.seed

    if manual_seed is not None:
        set_deterministic_environment(manual_seed, logger)

    train_loader, val_loader, test_loader = get_loaders(dataset_path, config)
    model.set_example(train_loader)

    trainer = Trainer(config=config)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = ParamsBuilder("Single-particle tracking data clssification train")
    parser.add_config_argument()
    parser.add_dataset_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
