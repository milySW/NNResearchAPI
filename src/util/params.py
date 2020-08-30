from argparse import ArgumentParser

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
