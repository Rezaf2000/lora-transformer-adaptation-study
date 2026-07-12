# Contributing

This is a personal learning project, but issues and pull requests are welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Before opening a pull request

- Run the tests with `pytest` and make sure they pass.
- Add or update a test when you change behaviour.
- Keep the style consistent with the surrounding code.

## Reporting a bug

Open an issue with what you expected, what happened, and the steps to reproduce it.
