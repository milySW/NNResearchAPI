from pathlib import Path
from types import SimpleNamespace
from typing import Union

import torch

from tqdm import tqdm

from configs import DefaultConfig
from src.loaders import DataLoader
from src.utils.checkers import image_folder
from src.utils.decorators import timespan
from src.utils.loaders import load_variable, load_y
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder

logger = get_logger("Evaluator")


def evaluate(
    config: Union[DefaultConfig, Path],
    input_path: Path,
    predict_path: Path,
    evaluate_path: Path,
    val_loader,
):
    if val_loader is None:
        _, val_loader, _ = DataLoader.get_loaders(input_path, config=config)

    predictions = load_y(predict_path)
    targets = [y for _, y in val_loader]
    data = [x for x, _ in val_loader]

    tensor_targets = targets[0]
    for tensor in targets[1:]:
        tensor_targets = torch.cat([tensor_targets, tensor])

    tensor_data = data[0]
    for tensor in data[1:]:
        tensor_data = torch.cat([tensor_data, tensor])

    targets = tensor_targets
    data = tensor_data

    for evaluator in config.evaluations.value_list():
        name = evaluator.name

        for prediction, target in tqdm(zip(predictions, targets), desc=name):
            evaluator(prediction, target)

        evaluator.manage_evals(
            predictions=predictions,
            targets=targets,
            data=data,
            output=evaluate_path,
            image=image_folder(val_loader.dataset),
        )


@timespan("Evaluation")
def main(
    config_path: Union[DefaultConfig, Path],
    input_path: Path,
    predict_path: Path,
    evaluate_path: Path,
    val_loader,
) -> Path:
    """
    Main function responsible for evaluation with passed model.

    Arguments:
        Path config_path: Path to main config (of :class:`DefaultConfig` class)
        Path input_path: Path to file with input data
        Path predict_path: Path to file with model predictions
        Path output_path: Path to output directory

    Returns:
        Path to the experiment root dir.
    """
    config = load_variable("config", config_path)

    evaluate(
        config=config,
        input_path=input_path,
        predict_path=predict_path,
        evaluate_path=evaluate_path,
        val_loader=val_loader,
    )
    return SimpleNamespace(root=evaluate_path.parent)


if __name__ == "__main__":
    parser = ParamsBuilder("Model evaluation")
    parser.add_config_argument()
    parser.add_input_argument()
    parser.add_predict_argument()
    parser.add_evaluate_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
