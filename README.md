# Maxwell 👹

[![PyPI
version](https://badge.fury.io/py/Maxwell.svg)](https://pypi.org/project/maxwell/)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/maxwell.svg)](https://pypi.org/project/maxwell/)
[![CircleCI](https://circleci.com/gh/CUNY-CL/maxwell.svg?style=svg&circle-token=43c60045a43c2b4d4e2ad95dce2968512e7fe8d6)](https://app.circleci.com/pipelines/github/CUNY-CL/maxwell?branch=main)

Maxwell is a Python library for learning the stochastic edit distance (SED)
between source and target alphabets for string transduction.

Given a corpus of source and target string pairs, it uses
expectation-maximization to learn the log-probability weights of edit actions
(copy, substitution, deletion, insertion) that minimize the number of edits
between source and target strings. These weights can then be used for edits over
unknown strings through Viterbi decoding.

## Install

First install dependencies:

    pip install -r requirements.txt

Then install:

    pip install .

It can then be imported like a regular Python module:

```python
import maxwell
```

## Usage

SED training can be done as either a command line tool or imported as a Python
dependency.

For command-line use, run:

    maxwell-train \
        --train /path/to/train/data \
        --output /path/to/output/file \
        --epochs "${NUM_EPOCHS}"

As a library object, you can use the `StochasticEditDistance` class to pass any
iterable of source-target pairs for training. Learned edit weights can then be
saved with the `write_params` method:

```python
from maxwell import sed


aligner = sed.StochasticEditDistance.fit_from_data(
    training_samples, NUM_EPOCHS
)
aligner.params.write_params("/path/to/output/file")
```

After training, parameters can be loaded from file to calculate optimal edits
between strings with the `action_sequence` method, which returns a tuple of the
learned optimal sequence and the weight given to the sequence:

```python
from maxwell import sed


params = sed.ParamsDict.read_params("/path/to/learned/parameters")
aligner = sed.StochasticEditDistance(params)
optimal_sequence, optimal_cost = aligner.action_sequence(source, target)
```

If only weight and no actions are required, `action_sequence_cost` can be called
instead:

```python
optimal_cost = aligner.action_sequence_cost(source, target)
```

Conversely, individual actions can be evaluated with the `action_cost` method:

```python
action_cost = aligner.action_cost(action)
```

## Details

### Data

The default data format is based on the SIGMORPHON 2017 shared tasks:

    source   target    ...

That is, the first column is the source (a lemma) and the second is the target.

In the case where the formatting is different, the `--source-col` and
`--target-col` flags can be invoked. For instance, for the SIGMORPHON 2016
shared task data format:

    source   ...    target

one would instead use the flag `--target-col 3` to use the third column as
target strings (note the use of 1-based indexing).

### Edit actions

Edit weights are maintained as a `ParamsDict` object, a dataclass comprising
three dictionaries and one floats. The dictionaries, and their indexing, are as
follows:

1.  `delta_sub` Keys: Tuple of source alphabet X target alphabet. Values:
    Substitution weight for all non-equivalent source-target pairs. If source
    symbol == target symbol, a seperate copy probability is used.
2.  `delta_del` Keys: All symbols in source string alphabet. Represents deletion
    from string. Values: Deletion weight for removal of source symbol from
    string.
3.  `delta_ins` Keys: All symbols in target string alphabet. Represents
    insertion into string. Values: Insertion weight for introduction of target
    symbol into string.
4.  `delta_eos` A float value representing probability of terminating the
    string.

In Python, these values may be accessed through a `StochasticEditDistance`
object's `params` attribute.

## References

Dempster, A., Laird, N., and Rubin, D. 1977. Maximum likelihood from incomplete
data via the EM algorithm. *Journal of the Royal Statistical Society, Series B*
30(1): 1-38.

Ristad, E. S. and Yianilos, P. N. 1998. Learning string-edit distance. *IEEE
Transactions on Pattern Analysis and Machine Intelligence* 20(5): 522-532.
