from pathlib import Path

from src.utils.decorators import timespan
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder
from tools.evaluate import main as evaluate
from tools.predict import main as predict
from tools.train import main as train

logger = get_logger("Pipeline")


@timespan("Training + Prediction + Evaluation")
def main(
    config_path: Path,
    dataset_path: Path,
    data_path: Path,
    predict_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:

    """
    Main function responsible for prediction with passed model.

    Arguments:
        Path config_path: Path to main config (of :class:`DefaultConfig` class)
        Path dataset_path: Path to dataset
        Path data_path: Path to file containing data
        Path predict_path: Path to file with model predictions
        Path input_path: Path to file with input data
        Path output_path: Path to output directory
    """

    logger.info("Trainer")
    model_path = train(config_path=config_path, dataset_path=dataset_path)
    root = model_path.parent.parent

    logger.info("Predictor")
    predict(
        config_path=config_path,
        input_path=input_path,
        model_path=model_path,
        predict_path=root / predict_path,
    )

    logger.info("Evaluator")
    evaluate(
        config_path=config_path,
        data_path=data_path,
        input_path=input_path,
        predict_path=root / predict_path,
        output_path=root / output_path,
    )


if __name__ == "__main__":
    parser = ParamsBuilder("Pipeline")
    parser.add_config_argument()
    parser.add_dataset_argument()
    parser.add_data_argument()
    parser.add_predict_argument()
    parser.add_input_argument()
    parser.add_output_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
