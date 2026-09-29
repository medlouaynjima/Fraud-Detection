#!/bin/bash

    # Function to create a development environment
    # This function installs Python 3.11.0 using pyenv, sets it as the local version,
    # installs Poetry, optionally initializes a new Poetry project or installs dependencies,
    # sets up an IPython kernel for Jupyter notebooks, and configures Poetry to use in-project virtual environments.
    #
    # Parameters:
    #   None
    #
    # Returns:
    #   None
    #
    # Example of usage:
    #   ./environment_setup.sh create_dev_env
    #
    # Notes:
    #   None
    pyenv install 3.11.0
    pyenv local 3.11.0
    python -m pip install --upgrade pip
    pip install poetry

    read -p "Do you want to initialize a new Poetry project? (y/n): " init_poetry

    if [ "$init_poetry" = "y" ]; then
        poetry init
        poetry add pandas
        poetry add joblib
        poetry add scikit-learn
        poetry add pytest --group dev
        poetry add pre-commit --group dev
        poetry add ipykernel --group dev
        poetry add pylint --group dev 
        poetry add pytest-cov --group dev
    elif [ "$init_poetry" = "n" ]; then
        poetry install
    fi

    poetry run python -m ipykernel install --user --name fraud-management

    poetry config virtualenvs.in-project true # This line enables Poetry to create a virtual environment (.venv folder) in the project's root directory rather than a central location
    poetry config virtualenvs.in-project # Checks and outputs the current value of the configuration option to serve as a confirmation step.
