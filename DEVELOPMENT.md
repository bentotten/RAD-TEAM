# Development
## Environments
See README.md in src/envs

## Tips

### IDEs

#### VSCode
- Point to the correct python interpreter:
    - Activate conda environment or preferred env
    - Get interpreters path
        - Open Python repl: `python3`
        - Get executable path: `import sys; sys.executable; exit()`
    - Point VSCode to the correct interpreter
        -  cmd/ctrl -> Shift-P -> Python: Select Interpreter -> Enter Interpreter Path -> [PATH from output]

### Testing
#### Pytest
- Coverage report can be found in `htmlcov/`
- Run a single test: `python3 -m pytest -v tests/test_testing.py`