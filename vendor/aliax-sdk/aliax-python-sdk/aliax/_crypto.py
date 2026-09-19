"""Pure-stdlib reader for the Aliax mapper container."""
from __future__ import annotations

import hashlib
import hmac
import struct

MAGIC = b"ALIAXM1\0"
SALT = b"aliax/v1"


def hkdf(key: bytes, info: bytes, length: int = 32) -> bytes:
    prk = hmac.new(SALT, key, hashlib.sha256).digest()
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()[:length]


def _qr(s, a, b, c, d):
    def rot(v, n):
        return ((v << n) | (v >> (32 - n))) & 0xFFFFFFFF
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF; s[d] = rot(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF; s[b] = rot(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF; s[d] = rot(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF; s[b] = rot(s[b] ^ s[c], 7)


def _block(key, counter, nonce):
    s = list(struct.unpack("<4I", b"expand 32-byte k"))
    s += list(struct.unpack("<8I", key)) + [counter] + list(struct.unpack("<3I", nonce))
    w = s[:]
    for _ in range(10):
        _qr(w, 0, 4, 8, 12); _qr(w, 1, 5, 9, 13); _qr(w, 2, 6, 10, 14); _qr(w, 3, 7, 11, 15)
        _qr(w, 0, 5, 10, 15); _qr(w, 1, 6, 11, 12); _qr(w, 2, 7, 8, 13); _qr(w, 3, 4, 9, 14)
    return struct.pack("<16I", *((w[i] + s[i]) & 0xFFFFFFFF for i in range(16)))


def chacha20(key, nonce, data, counter=1):
    out = bytearray(len(data))
    for off in range(0, len(data), 64):
        stream = _block(key, counter, nonce); counter += 1
        for i in range(min(64, len(data) - off)):
            out[off + i] = data[off + i] ^ stream[i]
    return bytes(out)


def open_container(container: bytes, asset_key: bytes) -> str:
    if len(container) < 60 or container[:8] != MAGIC or container[8] != 1:
        raise ValueError("invalid mapper container")
    body, tag = container[:-32], container[-32:]
    expected = hmac.new(hkdf(asset_key, b"mapper-mac"), body, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("mapper container integrity check failed")
    nonce = body[12:24]
    plain = chacha20(hkdf(asset_key, b"mapper-enc"), nonce, body[28:])
    if len(plain) != struct.unpack(">I", body[24:28])[0]:
        raise ValueError("mapper container length mismatch")
    return plain.decode("utf-8")
