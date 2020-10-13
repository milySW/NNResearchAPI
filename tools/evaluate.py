from pathlib import Path

from tqdm import tqdm

from src.utils.decorators import timespan
from src.utils.loaders import load_variable, load_x, load_y
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder

logger = get_logger("Evaluator")


@timespan("Evaluation")
def main(
    config_path: Path,
    data_path: Path,
    input_path: Path,
    predict_path: Path,
    evaluate_path: Path,
) -> None:
    """
    Main function responsible for evaluation with passed model.

    Arguments:
        Path config_path: Path to main config (of :class:`DefaultConfig` class)
        Path data_path: Path to file containing data
        Path input_path: Path to file with input data
        Path predict_path: Path to file with model predictions
        Path output_path: Path to output directory
    """
    config = load_variable("config", config_path)

    data = load_x(input_path)
    targets = load_y(data_path)
    predictions = load_y(predict_path)

    for evaluator in config.evaluations.value_list():
        name = evaluator.name

        for prediction, target in tqdm(zip(predictions, targets), desc=name):
            evaluator(prediction, target)

        evaluator.manage_evals(predictions, targets, data, evaluate_path)


if __name__ == "__main__":
    parser = ParamsBuilder("Model evaluation")
    parser.add_config_argument()
    parser.add_data_argument()
    parser.add_input_argument()
    parser.add_predict_argument()
    parser.add_evaluate_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args)

    main(**vars(args))
