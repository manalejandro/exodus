# Contributing to exodus

Thanks for helping build a free, non-profit compute network. exodus is a
small, deliberately simple codebase; the protocol doc ([docs/protocol.md](docs/protocol.md))
and incentives doc ([docs/incentives.md](docs/incentives.md)) are the design
reference.

## Setup

```bash
git clone https://github.com/manalejandro/exodus.git
cd exodus
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requirements: Python 3.10+.

## Checks

Before opening a pull request, run:

```bash
pytest -q                       # full suite (59 tests)
ruff check src tests            # lint (line length 88, target py310)
```

The test suite covers crypto, compute-unit accounting, the append-only ledger,
consensus (including a byzantine node test), rewards, the API, and a full
multi-node simulation that must converge on identical ledgers.

## Design conventions

- **Determinism is sacred.** Anything that goes into the ledger, or is derived
  from it (CUs, credits, sealer selection), must be a pure function of inputs
  that every node can verify identically. No wall-clock-dependent rules, no
  random tie-breaks, no mutable global state.
- **The ledger is append-only.** `ChainStore` never rewrites chain rows. Add
  migrations or a new table instead of editing history.
- **The core stays dependency-light.** The package runtime depends only on
  `pydantic`, `anyio`, `cryptography`, and `loguru`. Optional extras (`api`,
  `exo`) must be imported lazily inside functions so the core never requires
  them.
- **Messages are typed.** Protocol messages are Pydantic models with a 1:1
  topic mapping in `consensus/topics.py`.
- **No comments unless they earn their place.** Prefer a well-named helper or a
  docstring that explains *why*.

## How to add a feature

1. Add a failing test first (or a simulation case).
2. Implement it in the relevant module, keeping the protocol pure and
   deterministic.
3. Update the docs if the protocol, rewards, or API surface changes.
4. Run `pytest -q` and `ruff check src tests`, then open a PR.

## Licensing

By contributing you agree that your work is licensed under the same
Apache-2.0 terms as the rest of the project.
