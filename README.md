# Sed-align 🪀

Sed-align is a python library for learning the stochastic edit distance (SED)
and associated parameters for string transduction. Its primary use is for 
training the transducer model in the yoyodyne toolkit (LINK), but is viable
for any transduction problem.

Models are trained using expectation-maximization over a givne dataset.
See .....

## Install

First install dependencies:

    pip install -r requirements.txt

Then install:

    python setup.py install

Or:

    python setup.py develop

The latter creates a Python module in your environment that updates as you
update the code. It can then be imported like a regular Python module:

``` python
import yoyodyne
```

## Usage

For examples, see [`experiments`](experiments). See
[`train.py`](yoyodyne/train.py) and [`predict.py`](yoyodyne/predict.py) for all
model options.

## Details

The default data format is based on the SIGMORPHON 2017 shared tasks:

    source   target    feat1;feat2;...

That is, the first column is the source (a lemma), the second is the target (the
inflection), and the third contains semi-colon delimited feature strings.

For the SIGMORPHON 2016 shared task data format:

    source   feat1,feat2,...    target

one would instead specify `--target-col 3 --features-col 2 --features-sep ,`

Finally, to perform transductions without features (whether or not a feature
column exists in the data), one would simply specify `--features-col 0`.
