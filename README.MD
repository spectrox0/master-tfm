# master-tfm

`master-tfm` is a project for time series analysis using various Python tools.

## Requirements

- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- `Make` (Makefile) for running project scripts

## Installation

### Installing Make

#### macOS

Make is typically pre-installed on macOS. If it's missing, you can install it via Xcode Command Line Tools:

```bash
    xcode-select --install
```

Alternatively, you can use Homebrew:

```bash
   brew install make
```

#### Linux

Most Linux distributions come with Make pre-installed. If it's missing, you can install it using your package manager. For example, on Ubuntu:

```bash
sudo apt-get install build-essential
```

#### Windows

To use Make on Windows, you can install it via Chocolatey or use a Windows Subsystem for Linux (WSL):

Using Chocolatey:

    Install Chocolatey if you haven't already: https://chocolatey.org/install

    Install Make:

    bash

```powershell
choco install make
```

Using WSL:

    Install WSL: https://docs.microsoft.com/en-us/windows/wsl/install
    Use the Linux instructions above once WSL is installed.

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
```

## Downloading Data with initialize.py

To download the necessary data for this project, you need to run the initialize.py script, which is set up to download data automatically. Depending on your environment, you can do it in several ways:

```bash
poetry run download-data
```

On Linux/macOS (using a shell script):

```bash
./initialize.py
```

On Windows:
With PowerShell:

```powershell
./initialize.ps1
```

```cmd
initialize.cmd
```
