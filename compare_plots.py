from pathlib import Path

from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder

logger = get_logger("Compartor")


def main(root_path: Path, output_path: Path, experiments_collection: str):
    logger.info("Start generating plots")
    # READ DATA + MAKE PLOTS + SAVE TO PASSED DIRECTORY
    logger.info("End generating plots")


if __name__ == "__main__":
    parser = ParamsBuilder("Metrics results comparison plots")
    parser.add_root_argument()
    parser.add_output_argument()
    parser.add_experiments_collection_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args, delimeter=",")

    main(**vars(args))
