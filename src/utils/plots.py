from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.collections import flatten, ignore_none
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_name_from_names(names: List[str]) -> str:
    return "".join([i.replace("/", "[") + "]" for i in names.split(",")])


def read_or_none(path: Path) -> Optional[pd.DataFrame]:
    if path.is_file():
        return pd.read_csv(path)

    logger.warn(f"Path {path} does not exist.")
    return None


def get_columns(dataframe_dict: Dict[str, pd.DataFrame]) -> set:
    dataframes = ignore_none(list(dataframe_dict.values()))
    cols = [df.columns.tolist() for df in dataframes]

    return set(col for col in flatten(cols) if col != "epoch")


def save_column_plot(df_dict: Dict[str, pd.DataFrame], col: str, output: Path):
    plt.figure()

    for name, df in df_dict.items():
        if col in df.columns:
            plt.plot(df[col], label=name)

    plt.legend(loc="best")
    plt.title(col)

    plt.savefig(output / f"{col}.png", transparent=True)
    plt.close()


def save_columns_plot(df: pd.DataFrame, names: List[str], root_dir: Path):
    plt.figure()

    for name in names:
        plt.plot(df[name], label=name)

    plt.legend(loc="best")
    plt.title(names[0])

    plt.savefig(root_dir / f"{names[0]}.png", transparent=True)
    plt.close()


def save_csv_plots(data: Dict[str, Path], output: Path, prefix: str):
    dataframe_dict = {name: read_or_none(path) for name, path in data.items()}
    cols = get_columns(dataframe_dict)
    output.mkdir(parents=True, exist_ok=True)

    for col in cols:
        if not col.startswith(prefix):
            continue

        save_column_plot(dataframe_dict, col, output)


def save_data_plot(data: Iterable, path: Path):
    plt.figure()
    plt.plot(*data)

    plt.savefig(path, transparent=True)
    plt.close()
