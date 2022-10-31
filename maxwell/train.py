"""Training."""

import csv
from typing import List

import click

from . import sed, util


@click.command()
@click.option("--train-data-path", required=True)
@click.option("--source-col", type=int, default=1)
@click.option("--target-col", type=int, default=2)
@click.option("--source-sep", type=str, default="")
@click.option("--target-sep", type=str, default="")
@click.option("--output-path", required=True)
@click.option("--num-epochs", type=int, default=10)
def main(
    train_data_path,
    source_col,
    target_col,
    source_sep,
    target_sep,
    output_path,
    num_epochs,
):
    """Training.

    Args:
        train_data_path (_type_): _description_
        source_col (_type_): _description_
        target_col (_type_): _description_
        source_sep (_type_): _description_
        target_sep (_type_): _description_
        output_path (_type_): _description_
        num_epochs (_type_): _description_
    """
    util.log_info("Arguments:")
    for arg, val in click.get_current_context().params.items():
        util.log_info(f"\t{arg}: {val!r}")
    train_samples = get_samples(
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


def get_samples(
    filename: str,
    source_col: int,
    source_sep: str,
    target_col: int,
    target_sep: str,
):
    with open(filename, "r") as source:
        tsv_reader = csv.reader(source, delimiter="\t")
        for row in tsv_reader:
            source = _get_cell(row, source_col, source_sep)
            target = _get_cell(row, target_col, target_sep)
            # Adds third item for compatibility with yoyodyne dataloader.
            yield source, target


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


if __name__ == "__main__":
    main()
