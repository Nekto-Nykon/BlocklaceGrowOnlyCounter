"""
Cryptographic primitives used by the Blocklace.

Each node is identified by an ECDSA (P-256) key-pair.  A block identity is
the SHA-256 hash of the block content, signed with the creator's private key.
The signature binds content to creator in an unforgeable way (paper §2.1).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

# A node's public identity is its DER-encoded public key.
NodeId = bytes


# ── Key management ────────────────────────────────────────────────────────────

def generate_keypair() -> Tuple[ec.EllipticCurvePrivateKey, NodeId]:
    """Generate a fresh ECDSA key-pair; return (private_key, node_id)."""
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    return private_key, _pubkey_to_bytes(private_key.public_key())


def _pubkey_to_bytes(pub: ec.EllipticCurvePublicKey) -> NodeId:
    return pub.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _pubkey_from_bytes(data: bytes) -> ec.EllipticCurvePublicKey:
    return serialization.load_der_public_key(data, backend=default_backend())


# ── Signing / verification ────────────────────────────────────────────────────

def sign(data: bytes, private_key: ec.EllipticCurvePrivateKey) -> bytes:
    """Sign *data* with *private_key* using ECDSA-SHA256."""
    return private_key.sign(data, ec.ECDSA(hashes.SHA256()))


def verify_sig(data: bytes, signature: bytes, node_id: NodeId) -> bool:
    """Return True iff *signature* is a valid ECDSA-SHA256 signature of *data* by *node_id*."""
    try:
        _pubkey_from_bytes(node_id).verify(signature, data, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, Exception):
        return False


# ── Hashing / serialisation ───────────────────────────────────────────────────

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON serialisation (sorted keys, no extra whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
