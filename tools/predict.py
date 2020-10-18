from pathlib import Path

import torch

from tqdm import tqdm

from src.models import load_model
from src.models.utils import save_prediction
from src.utils.collections import batch_list
from src.utils.decorators import timespan
from src.utils.loaders import load_variable, load_x
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder
from src.utils.transforms import pred_transform

logger = get_logger("Predictor")


@timespan("Prediction")
def main(
    config_path: Path, input_path: Path, model_path: Path, predict_path: Path
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
    dtype = config.training.dtype

    x = load_x(input_path, dtype=dtype)
    batches = batch_list(x, config.prediction.batch_size)

    model = load_model(config, model_path)
    model.eval()

    all_preds = torch.tensor([])
    for input_data in tqdm(batches, desc="Predictions"):
        predictions = model(input_data)
        processed_preds = pred_transform(predictions, config.postprocessors)

        all_preds = torch.cat([all_preds, processed_preds])

    save_prediction(predictions=all_preds, output_path=predict_path)

    return predict_path.parent.parent


if __name__ == "__main__":
    parser = ParamsBuilder("Model prediction")
    parser.add_config_argument()
    parser.add_input_argument()
    parser.add_model_argument()
    parser.add_predict_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
