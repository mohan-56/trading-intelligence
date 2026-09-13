"""Crypto Intelligence backend.

Importing this package applies thread caps and TLS trust-store routing before
any numeric or networking library loads. Order matters -- see app/bootstrap.py.
"""

from app import bootstrap as _bootstrap  # noqa: F401  (import order is the point)

__version__ = "0.1.0"
