[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Automated testing](https://github.com/eikehmueller/mlconservation_code/actions/workflows/python-app.yml/badge.svg)](https://github.com/eikehmueller/mlconservation_code/actions/workflows/python-app.yml)

# Neural network solvers for differential equations with exact conservation laws

Implementation of Lagrangian neural networks with intrinsically built-in conservaion laws.

## Running the code
### Training the neural networks
The neural network models can be trained with the [train_model.py](src/train_model.py) script, which reads its parameters from a `.toml` configuration file. You might want to copy and modify the template in (training_parameters_template.toml)[training_parameters_template.toml]. To run the code, use

```
python src/train_model.py --parameterfile=PARAMETERFILE
```

where `PARAMETERFILE` is the name of the `.toml` file with the parameters. If you leave out the `--parameterfile` flag, this defaults to `training_parameters.toml`.

### Evaluating and visualising trained models

## Testing
Tests are collected in the [tests subdirectory](tests). To run all tests use

```
pytest
```

