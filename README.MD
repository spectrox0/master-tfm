# master-tfm

`master-tfm` is a project for time series analysis using various Python tools.

## Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

## Installation

### Using Poetry

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/master-tfm.git
    cd master-tfm
    ```

2. Install dependencies:

    ```bash
    poetry install
    ```

3. Activate the virtual environment:

    ```bash
    poetry config virtualenvs.in-project true && poetry shell
    ```

4. Generate the `requirements.txt` and `requirements-dev.txt` files:

    ```bash
    make generate_requirements_with_poetry
    ```

### Using requirements-dev.txt

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/master-tfm.git
    cd master-tfm
    ```

2. Create and activate a virtual environment in `.venv`:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Install the development dependencies:

    ```bash
    pip install -r requirements-dev.txt
    ```

4. Generate the `requirements.txt` and `requirements-dev.txt` files:

    ```bash
    make generate_requirements
    ```

## Usage

After activating the virtual environment and installing all dependencies, you can start working on your project.

## Available Scripts

You can use the following commands to generate the `requirements.txt` files and activate the virtual environment:

```bash
make generate_requirements
