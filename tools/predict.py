from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

from torch.nn import Module
from tqdm import tqdm

from src.loaders import DataLoader
from src.models import load_model
from src.models.utils import save_prediction
from src.utils.checkers import image_folder
from src.utils.decorators import timespan
from src.utils.loaders import load_variable
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder
from src.utils.transforms import input_transform, pred_transform

logger = get_logger("Predictor")


@timespan("Prediction")
def main(
    config_path: Path,
    input_path: Path,
    model_path: Path,
    predict_path: Path,
    val_loader: Optional[DataLoader] = None,
) -> Path:
    """
    Main function responsible for prediction with passed model.

    Arguments:
        Path config_path: Path to main config (of :class:`DefaultConfig` class)
        Path input_path: Path to file with input data
        Path model_path: Path to trained model
        Path predict_path: Path to output directory

    Returns:
        Path to the experiment root dir.
    """

    config = load_variable("config", config_path)

    if val_loader is None:
        pipeline = False
        _, val_loader, _ = DataLoader.get_loaders(input_path, config=config)

    else:
        pipeline = True

    is_image_folder = image_folder(val_loader.dataset)

    model: Module = load_model(config, model_path)
    model.eval()

    all_preds = torch.tensor([])
    for inputs, labels in tqdm(val_loader, desc="Predictions"):

        if not is_image_folder and not pipeline:
            inputs, _ = input_transform(
                input_data=inputs,
                input_labels=labels,
                preprocessors=config.preprocessors,
            )

        inputs = inputs.permute(0, 1, 3, 2)
        inputs = inputs.squeeze()

        inputs = inputs[[all(x != torch.tensor([0, 0])) for x in inputs]]
        inputs = inputs.unsqueeze(dim=0).unsqueeze(dim=0)
        inputs = inputs.permute(0, 1, 3, 2)

        predictions = model(inputs)
        processed_preds = pred_transform(predictions, config.postprocessors)

        all_preds = torch.cat([all_preds, processed_preds])

    save_prediction(predictions=all_preds, output_path=predict_path)

    return SimpleNamespace(root=predict_path.parent.parent)


if __name__ == "__main__":
    parser = ParamsBuilder("Model prediction")
    parser.add_config_argument()
    parser.add_input_argument()
    parser.add_model_argument()
    parser.add_predict_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
