from pathlib import Path
from typing import Optional

from src.utils.decorators import timespan
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder
from tools.evaluate import main as evaluate
from tools.predict import main as predict
from tools.train import main as train

logger = get_logger("Pipeline")


@timespan("Pipeline")
def main(
    config_path: Path,
    dataset_path: Path,
    predict_path: Path,
    input_path: Path,
    evaluate_path: Optional[Path] = None,
) -> None:

    """
    Main function responsible for prediction with passed model.

    Arguments:
        Path config_path: Path to main config (of :class:`DefaultConfig` class)
        Path dataset_path: Path to dataset
        Path predict_path: Path to file with model predictions
        Path input_path: Path to file with input data
        Path evaluate_path: Path to evaluations
    """

    logger.info("Trainer")
    result = train(config_path=config_path, dataset_path=dataset_path)

    info = "Pipline required training_config.py with attribute save = True"
    assert len(vars(result)) != 1, info

    root = result.model_path.parent.parent

    logger.info("Predictor")
    predict(
        config_path=config_path,
        input_path=input_path,
        model_path=result.model_path,
        predict_path=root / predict_path,
        val_loader=result.val_loader,
    )

    if evaluate_path:
        logger.info("Evaluator")
        evaluate(
            config_path=config_path,
            input_path=input_path,
            predict_path=root / predict_path,
            evaluate_path=root / evaluate_path,
            val_loader=result.val_loader,
        )


if __name__ == "__main__":
    parser = ParamsBuilder("Pipeline")
    parser.add_config_argument()
    parser.add_dataset_argument()
    parser.add_predict_argument()
    parser.add_input_argument()
    parser.add_evaluate_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
