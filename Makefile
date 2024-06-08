SHELL=/bin/bash
LINT_PATHS=src/ tests/ setup.py

# Auto-format files
format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

test:
	./runtests

# Linting 
lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

# Type Checking
type:
	mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports ${LINT_PATHS}

# Pre-checks 
commit-checks: check-codestyle type lint