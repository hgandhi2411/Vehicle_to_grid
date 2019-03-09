### Set-up Virtual Environment for Testing


Run the following once:
``sh
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

### Running Tests
To run tests with coverage: `pytest --cov=v2g`

### Running optimzer
Run `v2g-optimize` in any directory after installing
