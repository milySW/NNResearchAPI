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
    """
    Function responsible for saving plots containing
    info about metric values in every experiment.

    Arguments:
        Path root_path: Path to the root folder for the subprocess
        Path output_path: Path to output directory
        str experiments: Relative paths to experiments separated by a coma
        str supported_files: Names of files necessary for tool usage
        str prefix: Prefix for data taken into account
    """

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
