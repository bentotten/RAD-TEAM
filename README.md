# RAD-TEAM 
[Multi-Agent Deep Reinforcement Learning for
Radiation Localization Radiation Localization](https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=7590&context=open_access_etds) by Ben Totten

## Notice
This repository is undergoing a major refactor. For thesis code, see the Masters thesis branch.

## Prerequisites 
- [Optional] Environment manager: 
    - [Recommended] [Micromamba](https://mamba.readthedocs.io/en/latest/installation.html)

## Setup
1. Initialize environment 
    1. [Option 1] with environment manager

        `micromamba create --file environment.yml && micromamba activate rad-team`


    1. [Option 2] with pip

        `pip install -e .`

## Training
Dont forget to activate any python environments
    `micromamba activate rad-team`

## Testing
Run all tests with `make test`

## Clean
Refresh with `make clean`

## Docs
Generate docs with `make docs`. These will generate in the docs/build/html folder.

## Development
See DEVELOPMENT.md

## Structure

### Utils
This directory contains useful tools including
- normalizers
- standardizers
