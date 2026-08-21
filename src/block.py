"""
Block and BlockId data structures — §2.2 of the Blocklace paper.

  Block  b = (identity i, content C)
  C      = (payload v, predecessors P, known_byz K)
  i      = SHA-256(C) signed by creator's private key

known_byz encodes which Byzantine nodes were known to the creator at the
time of block creation — mirrors the TLA+ field known_byz in each block.
This enables the CanAccept / Byzantine-repelling check: a block from node n
is accepted only if byz[receiver] ⊆ b.known_byz (or it is first evidence).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet

from .crypto import NodeId, canonical_bytes, sha256, sign, verify_sig


# ── BlockId ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlockId:
    """
    Block identity i = signedhash(C, k_p).

    content_hash  SHA-256 of the serialised block content
    signature     ECDSA-SHA256 of content_hash with the creator's private key
    node_id       DER-encoded public key of the creator
    """
    content_hash: bytes
    signature: bytes
    node_id: NodeId

    def to_hex_dict(self) -> dict:
        return {
            "h": self.content_hash.hex(),
            "s": self.signature.hex(),
            "n": self.node_id.hex(),
        }

    @classmethod
    def from_hex_dict(cls, d: dict) -> "BlockId":
        return cls(
            content_hash=bytes.fromhex(d["h"]),
            signature=bytes.fromhex(d["s"]),
            node_id=bytes.fromhex(d["n"]),
        )

    def short(self) -> str:
        return self.content_hash.hex()[:8]


# ── BlockContent ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlockContent:
    """
    C = (v, P, K) in our extended model.

    payload       arbitrary JSON-serialisable application data
    predecessors  frozenset of BlockIds (frontier at creation time)
    known_byz     frozenset of NodeIds known to be Byzantine at creation time
                  (mirrors TLA+ known_byz field — enables CanAccept check)
    """
    payload: Any
    predecessors: FrozenSet[BlockId]
    known_byz: FrozenSet[NodeId] = field(default_factory=frozenset)

    def to_bytes(self) -> bytes:
        preds_sorted = sorted(
            [bid.to_hex_dict() for bid in self.predecessors],
            key=lambda d: d["h"],
        )
        known_byz_sorted = sorted(nid.hex() for nid in self.known_byz)
        return canonical_bytes({
            "payload": self.payload,
            "predecessors": preds_sorted,
            "known_byz": known_byz_sorted,
        })


# ── Block ─────────────────────────────────────────────────────────────────────

@dataclass
class Block:
    """b = (i, C) — identity paired with content."""
    identity: BlockId
    content: BlockContent

    @property
    def node_id(self) -> NodeId:
        return self.identity.node_id

    @property
    def predecessors(self) -> FrozenSet[BlockId]:
        return self.content.predecessors

    @property
    def payload(self) -> Any:
        return self.content.payload

    @property
    def known_byz(self) -> FrozenSet[NodeId]:
        """Byzantine nodes known to the creator at creation time (TLA+ known_byz)."""
        return self.content.known_byz

    def has_valid_signature(self) -> bool:
        expected_hash = sha256(self.content.to_bytes())
        if expected_hash != self.identity.content_hash:
            return False
        return verify_sig(self.identity.content_hash, self.identity.signature, self.identity.node_id)

    @classmethod
    def create(
        cls,
        payload: Any,
        predecessors: FrozenSet[BlockId],
        private_key,
        node_id: NodeId,
        known_byz: FrozenSet[NodeId] = frozenset(),
    ) -> "Block":
        """new_p(B, v) — create and sign a fresh block, embedding known_byz."""
        content = BlockContent(
            payload=payload,
            predecessors=predecessors,
            known_byz=known_byz,
        )
        content_hash = sha256(content.to_bytes())
        signature = sign(content_hash, private_key)
        identity = BlockId(content_hash=content_hash, signature=signature, node_id=node_id)
        return cls(identity=identity, content=content)

    def __repr__(self) -> str:
        return f"Block({self.identity.short()} by {self.node_id.hex()[:8]})"
