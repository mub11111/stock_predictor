# Claude Code Guidelines — Stock Predictor (A股智能预测系统)

## Project Context
- Python 3.10+, PyQt6 GUI, PyTorch models, PyQtGraph charts
- Entry point: `python main.py` (in `stock_predictor/`)
- Structure: `gui/` (UI), `model/` (ML), `data/` (preprocessing/rules), `storage/` (SQLite)
- Config: `config.py` (AppConfig dataclass)

## Coding Guidelines
- **Minimal diffs**: Only change what's requested. No "while I'm here" refactoring.
- **Simple first**: Write the simplest code that works. No premature abstraction.
- **Clarify before implementing**: Ask questions if requirements are unclear.
- **No blind refactoring**: Don't reorganize code or rename things unless asked.
- **Single-line fixes**: For trivially obvious bugs, fix directly without extensive planning.
- **Existing patterns**: Follow the conventions already used in the file being edited.

## Critical Safety
- **Never commit secrets**: .env files, API keys, credentials
- **Never run destructive commands without confirmation**: git push --force, rm -rf, etc.
- **Verify compilation**: Run `python -c "import py_compile; py_compile.compile('file.py', doraise=True)"` after edits

## Trade-offs
- Bias toward correctness over speed for non-trivial work
- For trivial tasks (typos, one-liners), use judgment — don't over-plan
- When in doubt, ask before acting
