import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path


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

    def add_model_argument(
        self, info: str = "Path to model saved in file"
    ) -> ArgumentParser:
        self.add_argument(
            "--model_path", type=Path, help=info, required=False,
        )

    @staticmethod
    def log_parser(args: Namespace, width: int = 20):
        def l_align(string: str, width=width) -> str:
            return str(string).ljust(width)

        args_items = args.__dict__.items()
        args = [f"{l_align(k)}: {l_align(v)} \n" for k, v in args_items]
        sys.stdout.write("\nParsed arguments:\n")
        sys.stdout.write(f"{''.join(args)} \n")
