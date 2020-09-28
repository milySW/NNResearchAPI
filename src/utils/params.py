import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any


class ParamsBuilder(ArgumentParser):
    def __init__(self, description: str):
        super().__init__(description=description)

    def add_config_argument(
        self, info: str = "Path to .py config file"
    ) -> ArgumentParser:
        self.add_argument(
            "--config_path", type=Path, help=info, required=True,
        )

    def add_dataset_argument(
        self, info: str = "Path to dataset directory"
    ) -> ArgumentParser:
        self.add_argument(
            "--dataset_path", type=Path, help=info, required=False,
        )

    def add_output_argument(
        self, info: str = "Path to output directory"
    ) -> ArgumentParser:
        self.add_argument(
            "--output_path", type=Path, help=info, required=True,
        )

    def add_root_argument(
        self, info: str = "The path to the root folder for the subprocess"
    ) -> ArgumentParser:
        self.add_argument(
            "--root_path", type=Path, help=info, required=True,
        )

    def add_model_argument(
        self, info: str = "Path to model saved in file"
    ) -> ArgumentParser:
        self.add_argument(
            "--model_path", type=Path, help=info, required=False,
        )

    def add_experiments_collection_argument(
        self, info: str = "List of relative paths to experiments"
    ) -> ArgumentParser:
        self.add_argument(
            "--experiments_collection", type=str, help=info, required=True,
        )

    @staticmethod
    def log_parser(args: Namespace, width: int = 20, delimeter: str = ""):
        def l_align(string: str, width=width) -> str:
            return str(string).ljust(width)

        def listed(var: Any):
            if delimeter and isinstance(var, str):
                var = "".join(["\n- ", var.replace(delimeter, "\n-")])
            return var

        items = args.__dict__.items()
        args = [f"{l_align(k)}: {l_align(listed(v))} \n" for k, v in items]
        sys.stdout.write("\nParsed arguments:\n")
        sys.stdout.write(f"{''.join(args)} \n")
