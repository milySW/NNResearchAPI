from pathlib import Path

from src.utils.collections import unpack_paths
from src.utils.logging import get_logger
from src.utils.params import ParamsBuilder
from src.utils.plots import get_name_from_names, save_csv_plots

logger = get_logger("Comparator")


def main(
    root_path: Path,
    output_path: Path,
    experiments: str,
    supported_files: str,
    prefix: str,
):
    logger.info("Start generating plots")
    output = output_path / get_name_from_names(experiments)
    experiments = unpack_paths(root_path, experiments, supported_files)

    for group in experiments:
        save_csv_plots(data=group, output=output, prefix=prefix)

    logger.info("End generating plots")


if __name__ == "__main__":
    parser = ParamsBuilder("Metrics results comparison plots")
    parser.add_root_argument()
    parser.add_output_argument()
    parser.add_prefix_argument()
    parser.add_supported_files_argument()
    parser.add_experiments_argument()

    args, _ = parser.parse_known_args()
    parser.log_parser(args, delimeter=",")

    main(**vars(args))
