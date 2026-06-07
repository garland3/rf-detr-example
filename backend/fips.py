"""FIPS compatibility shim for MD5 used as a non-security checksum.

On hosts where OpenSSL runs in FIPS mode, MD5 is disabled outright and any
call to ``hashlib.md5()`` raises::

    ValueError: [digital envelope routines: EVP_DigestInit_ex] disabled for FIPS

RF-DETR uses MD5 purely to verify the integrity of its downloaded pretrained
weights (a checksum, not a security operation). Python supports exactly this
case via the ``usedforsecurity=False`` flag, which OpenSSL permits even under
FIPS. Upstream calls ``hashlib.md5()`` without that flag, so model startup
fails on FIPS systems.

``enable()`` wraps ``hashlib.md5`` (and ``hashlib.new("md5", ...)``) so the
flag is set automatically. It is a no-op on non-FIPS hosts and idempotent.
"""
from __future__ import annotations

import hashlib

_PATCHED = False


def _md5_works() -> bool:
    try:
        hashlib.md5(b"")
        return True
    except ValueError:
        return False


def enable() -> bool:
    """Make ``hashlib.md5`` usable under FIPS for non-security checksums.

    Returns ``True`` if a patch was applied (i.e. the host blocks plain MD5),
    ``False`` if MD5 already worked and nothing was changed.
    """
    global _PATCHED
    if _PATCHED or _md5_works():
        return False

    _orig_md5 = hashlib.md5
    _orig_new = hashlib.new

    def _md5(data=b"", *, usedforsecurity=False):
        return _orig_md5(data, usedforsecurity=usedforsecurity)

    def _new(name, data=b"", *, usedforsecurity=True):
        if name.lower() == "md5":
            usedforsecurity = False
        return _orig_new(name, data, usedforsecurity=usedforsecurity)

    hashlib.md5 = _md5  # type: ignore[assignment]
    hashlib.new = _new  # type: ignore[assignment]
    _PATCHED = True
    return True
