# Stochastic Models for Vehicle-to-Grid Microeconomics

## Installation, Usage and Testing

### Set-up Virtual Environment for Testing

To set-up with a virtual environment, run the following once:
```sh
pip install --user virtualenv
cd /path/to/code
virtualenv venv
pip install -e .
```

Each time you want to work with code, run:
```sh
source venv/bin/activate
....
#when done
deactivate
```
### Set-up without virtual environment

To install normally, just run `pip install -e .` from the 
root directory.

### Running Tests
To run tests with coverage run `pytest --cov=v2g` from the root directory.

### Running optimzer
Run `v2g-optimize` in any directory after installing to use the optimzer.

## Citing
To cite the code, use this citation:
```
@misc{gandhi2021citywide,
      title={City-wide modeling of Vehicle-to-Grid Economics to Understand Effects of Battery Performance}, 
      author={Heta A. Gandhi and Andrew D. White},
      year={2021},
      eprint={2108.05837},
      archivePrefix={arXiv},
      primaryClass={stat.AP}
}
```

### Disclaimer
This repository is not actively maintained. 
