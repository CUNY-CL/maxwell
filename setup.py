import os

import setuptools

if getattr(setuptools, "__version__", "0") < "39":
    # v36.4.0+ needed to automatically include README.md in packaging
    # v38.6.0+ needed for long_description_content_type in setup()
    raise EnvironmentError(
        "Your setuptools is too old. "
        "Please run 'pip install --upgrade pip setuptools'."
    )


_THIS_DIR = os.path.dirname(os.path.realpath(__file__))


with open(os.path.join(_THIS_DIR, "README.md")) as f:
    _LONG_DESCRIPTION = f.read().strip()


__version__ = "0.2.0.post1"


def main() -> None:
    setuptools.setup(
        name="maxwell",
        version=__version__,
        description="Stochastic Edit Distance aligner for string transduction",
        long_description=_LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author="""Simon Clematide, Peter Makarov,
                    Travis M. Bartley""",
        keywords=[
            "computational linguistics",
            "morphology",
            "natural language processing",
            "language",
        ],
        classifiers=[
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Topic :: Text Processing :: Linguistic",
        ],
        license="Apache 2.0",
        python_requires=">=3.9",
        install_requires=[
            "click>=8.1.3",
            "numpy>=1.20.1",
            "scipy>=1.6",
            "tqdm>=4.64.1",
        ],
        packages=setuptools.find_packages(),
        entry_points={
            "console_scripts": [
                "maxwell-train = maxwell.train:main",
            ]
        },
        data_files=[(".", ["LICENSE.txt"])],
    )


if __name__ == "__main__":
    main()
