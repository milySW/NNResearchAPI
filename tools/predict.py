from pathlib import Path

from src.models import load_model
from src.models.utils import save_prediction
from src.utils.decorators import timespan
from src.utils.loaders import load_variable, load_x
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder

logger = get_logger("Predictor")


@timespan("Prediction")
def main(
    config_path: Path, input_path: Path, model_path: Path, predict_path: Path
) -> None:
    """
    Main function responsible for prediction with passed model.

    Arguments:
        Path config_path: Path to main config (of :class:`DefaultConfig` class)
        Path input_path: Path to file with input data
        Path model_path: Path to trained model
        Path predict_path: Path to output directory
    """
    config = load_variable("config", config_path)
    dtype = config.training.dtype

    model = load_model(config, model_path)
    x = load_x(input_path, dtype=dtype)

    model.eval()
    save_prediction(predictions=model(x), output_path=predict_path)


if __name__ == "__main__":
    parser = ParamsBuilder("Model prediction")
    parser.add_config_argument()
    parser.add_input_argument()
    parser.add_model_argument()
    parser.add_predict_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
