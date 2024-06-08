# RAD-TEAM 

## Notice
This repository is undergoing a major refactor. For thesis code, see the Masters thesis branch.

## Prerequisites 
- [Optional] Environment manager: 
    - [Recommended] [Micromamba](https://mamba.readthedocs.io/en/latest/installation.html)

## Setup
1. Initialize environment 
    1. [Option 1] with environment manager

        `micromamba create --file environment.yml`

    1. [Option 2] with pip

        `pip install -e .`

1. Activate environment
    `micromamba activate rad-team`

## Testing
Run all tests with `make test`

## Clean
Refresh with `make clean`

## Docs
Generate docs with `make docs`. These will generate in the docs/build/html folder.

## Development
See DEVELOPMENT.md