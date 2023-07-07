"""Stochastic edit distance training."""

import argparse
import csv
from typing import Iterator, List, Tuple

from . import sed, util


def _get_cell(row: List[str], col: int, sep: str) -> List[str]:
    """Returns the split cell of a row.
    Args:
        row (List[str]): the split row.
        col (int): the column index
        sep (str): the string to split the column on; if the empty string,
            the column is split into characters instead.
    Returns:
        A list of symbols from that cell.
    """
    cell = row[col - 1]  # -1 because we're using one-based indexing.
    return list(cell) if not sep else cell.split(sep)


def _get_samples(
    filename: str,
    source_col: int,
    source_sep: str,
    target_col: int,
    target_sep: str,
) -> Iterator[Tuple[str, str]]:
    with open(filename, "r") as source:
        tsv_reader = csv.reader(source, delimiter="\t")
        for row in tsv_reader:
            source = _get_cell(row, source_col, source_sep)
            target = _get_cell(row, target_col, target_sep)
            yield source, target


def main(
    train_data_path,
    source_col,
    target_col,
    source_sep,
    target_sep,
    output_path,
    num_epochs,
):
    """Training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source_col", type=int, default=1)
    parser.add_argument("--target_col", type=int, default=2)
    parser.add_argument("--source_sep", type=str, default="")
    parser.add_argument("--target_sep", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parser_args()
    util.log_info("Arguments:")
    for arg, val in vars(args).items():
        util.log_info(f"\t{arg}: {val!r}")
    train_samples = _get_samples(
        train_data_path,
        source_col,
        source_sep,
        target_col,
        target_sep,
    )
    sed_aligner = sed.StochasticEditDistance.fit_from_data(
        train_samples, epochs=num_epochs
    )
    sed_aligner.params.write_params(output_path)


if __name__ == "__main__":
    main()
