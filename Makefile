SHELL=/bin/bash
LINT_PATHS=src/ tests/ setup.py

# Cleans
.PHONY: clean clean-build clean-pyc clean-test 

clean: clean-build clean-pyc clean-test clean-docs

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	find . -name '.coverage' -exec rm -r {} +
	rm -fr htmlcov/
	rm -fr .pytest_cache/
	rm -fr .mypy_cache/
	rm -fr .ruff_cache/

clean-docs:
	cd docs && make clean
	rm -fr docs/_build

.PHONY: test format lint check-codestyle type commit-checks

# Run pytests
test:
	./runtests

# Linting 
format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero
	@echo 'If issues, try make format'

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

# Type Checking
type:
	mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports ${LINT_PATHS}

# Pre-checks 
commit-checks: test lint type check-codestyle

.PHONY: docs spelling

# Docs
docs:
	cd docs && make html

spelling:
	cd docs && make spelling

.PHONY: egg release dist install

egg:
	python setup.py egg_info

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean
	micromamba create --file environment.yml
