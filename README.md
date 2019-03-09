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

