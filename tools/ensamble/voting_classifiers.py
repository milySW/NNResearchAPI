from pathlib import Path

import torch

from tqdm import tqdm

from src.loaders import DataLoader
from src.models import load_model
from src.models.utils import save_prediction
from src.utils.checkers import set_param
from src.utils.collections import unpack_string
from src.utils.decorators import timespan
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder
from src.utils.transforms import pred_transform
from tools.ensamble.utils import get_config
from tools.ensamble.vote import two_classifiers
from tools.evaluate import evaluate

logger = get_logger("Predictor")


@timespan("Two classifiers")
def main(
    root_path: Path,
    experiments: str,
    model_paths: str,
    input_path: Path,
    predict_path: Path,
    config_path: Path,
    evaluate_path: Path,
) -> Path:
    """
    Main function responsible for prediction with passed model.

    Arguments:
        Path root_path: Path to the root folder for the subprocess
        str experiment_paths:  Relative paths to experiments separated by coma
        Path model_path: Subpath to trained model
        Path input_path: Path to file with input data
        Path predict_path: Path to output directory
        Path config_path: Path to main config (of :class:`DefaultConfig` class)
        Path evaluate_path: Path to evaluations

    Returns:
        Path to the experiment root dir.
    """
    models_with_config = []

    data_type = None
    prediction_bs = None
    main_config = None

    zipped = zip(unpack_string(experiments), unpack_string(model_paths))

    for index, (experiment, model_path) in enumerate(zipped):
        config_paths = (root_path / experiment / config_path).iterdir()
        config = get_config(config_paths)

        if index == 0:
            main_config = config

        model = load_model(config, root_path / experiment / model_path)
        model.eval()

        models_with_config.append(dict(model=model, config=config))

        data_type = set_param(
            previous=data_type, current=config.training.dtype, name="dtype"
        )

        prediction_bs = set_param(
            previous=prediction_bs,
            current=config.prediction.batch_size,
            name="batch_size",
        )

    _, val_loader, _ = DataLoader.get_loaders(input_path, config=main_config)

    model_preds = []

    for model_with_config in models_with_config:
        model = model_with_config["model"]
        config = model_with_config["config"]

        all_preds = torch.tensor([])
        for x, _ in tqdm(val_loader, desc="Predictions"):
            predictions = model(x)
            processed_preds = pred_transform(
                preds=predictions, postprocessors=config.postprocessors
            )

            all_preds = torch.cat([all_preds, processed_preds])

        model_preds.append(all_preds)

    predictions = two_classifiers(model_preds, loader=val_loader)
    save_prediction(
        predictions=predictions, output_path=root_path / predict_path,
    )

    if evaluate_path:
        logger.info("Evaluator")
        evaluate(
            config=main_config,
            input_path=input_path,
            predict_path=root_path / predict_path,
            evaluate_path=root_path / evaluate_path,
            val_loader=val_loader,
        )


if __name__ == "__main__":
    parser = ParamsBuilder("Voting classifier: Prediction + Evaluation")
    parser.add_root_argument()
    parser.add_experiments_argument()
    parser.add_models_argument()
    parser.add_input_argument()
    parser.add_predict_argument()
    parser.add_config_argument()
    parser.add_evaluate_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
