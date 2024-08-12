generate_requirements:
	pip-compile --output-file=requirements.txt requirements.in
	pip-compile --output-file=requirements-dev.txt requirements-dev.in

generate_requirements_with_poetry:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --dev



activate_virtualenv_poetry:
	poetry config virtualenvs.in-project true && poetry shell

activate_virtualenv:
	source .venv/bin/activate




