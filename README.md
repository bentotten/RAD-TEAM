# RAD-TEAM 

## Notice
This repository is undergoing a major refactor. For thesis code, see the Masters thesis branch.

## Prerequisites 
- [Optional] Environment manager: 
    - [Recommended] [Micromamba](https://mamba.readthedocs.io/en/latest/installation.html)

## Setup
1. Initialize environment 
    a. With micromamba
        `micromamba create --file env.yml`

    b. With pip
        `pip install -e .`

1. Activate environment
    `micromamba activate rad-team`

## Testing
Run all tests with `make test`

## Development
See DEVELOPMENT.md