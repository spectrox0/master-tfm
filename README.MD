python -m venv .venv
source .venv/activate/bin
python -m pip install poetry
poetry install
pip install

<!-- // poetry init -->

pip install ipykernel

# install jupyter lab

pip install jupyterlab

# nbdime install with pip

pip install --upgrade nbdime

# config git

# default is current project

nbdime config-git --enable --global

# poetry

poetry add nbdime

# display execution time of each cell

pip install jupyterlab_execute_time

# show interactive matplotlib plots

pip install ipympl

pipenv install jupyterlab-lsp || pip install jupyterlab-lsp
pipenv install python-lsp-server[all]
mkdir .lsp_symlink && cd .lsp_symlink && ln -s /home home

# collapsable headings

pipenv install aquirdturtle_collapsible_headings || pip install aquirdturtle_collapsible_headings

# table of contents

jupyter labextension install @jupyterlab/toc

python -m ipykernel install --user --name poetry_env --display-name "Poetry Environment"
