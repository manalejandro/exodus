"""
exodus — a free, non-profit, open distributed compute network.

exodus builds on the `exo` project to let anyone share their idle compute
(GPU/CPU/RAM) with a global network that runs AI models for free.  Nodes agree
on who contributed what via a lightweight distributed consensus protocol
("Proof-of-Contribution"), record the agreement in an append-only, hash-chained
ledger, and reward contributors with *extra AI time* — priority scheduling and
concurrency quota on the shared pool.  No money, no tokens, no ads.
"""

from exodus.version import __version__

__all__ = ["__version__"]
