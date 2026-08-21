"""
Byzantine node behaviours — exact translation of TLA+ Newtry actions.

ByzantineEquivocate(bz):
  - Creates blockA (nonce N) and blockB (nonce N+1) with same parents
  - Stores BOTH in B[bz]
  - Sends BOTH to all Nodes \\ {bz}
  - known_byz = byz[bz] at time of creation
  - UNCHANGED byz  (Byzantine node never updates its own byz)

ByzantineAddHonest(bz, payload):
  - Creates one block like Add
  - Stores in B[bz]
  - Sends only to Honest nodes (not to other Byzantine)
  - UNCHANGED byz
"""
from __future__ import annotations

from typing import Any, FrozenSet, Tuple

from .block import Block, BlockContent, BlockId
from .blocklace import Blocklace
from .crypto import NodeId, generate_keypair, sha256, sign


class EquivocatingNode:
    """
    Byzantine equivocating node.

    Maintains B[bz] (own blocklace) which stores both equivocating blocks,
    exactly as TLA+: B' = [B EXCEPT ![bz] = B[bz] ∪ {blockA, blockB}]

    byz[bz] is never updated — Byzantine nodes do not track Byzantine peers.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.private_key, self.node_id = generate_keypair()
        # B[bz] in TLA+
        self.blocklace: Blocklace = Blocklace()
        # byz[bz] = {} always (UNCHANGED byz in TLA+)
        self.byz: FrozenSet[NodeId] = frozenset()

    def equivocate(
        self,
        payload_a: Any,
        payload_b: Any,
        predecessors: FrozenSet[BlockId],
    ) -> Tuple[Block, Block]:
        """
        ByzantineEquivocate(bz) — create two incomparable blocks.

        Both share the same predecessors (parents) and known_byz = byz[bz] = {}.
        Both stored in B[bz].  Both returned for the network to deliver.
        """
        block_a = Block.create(payload_a, predecessors, self.private_key, self.node_id, self.byz)
        block_b = Block.create(payload_b, predecessors, self.private_key, self.node_id, self.byz)
        # B' = B[bz] ∪ {blockA, blockB}
        self.blocklace._blocks[block_a.identity] = block_a.content
        self.blocklace._blocks[block_b.identity] = block_b.content
        return block_a, block_b

    def add_honest(
        self,
        payload: Any,
        predecessors: FrozenSet[BlockId],
    ) -> Block:
        """
        ByzantineAddHonest(bz, payload) — one normal block, sent only to Honest.
        Stored in B[bz].
        """
        block = Block.create(payload, predecessors, self.private_key, self.node_id, self.byz)
        self.blocklace._blocks[block.identity] = block.content
        return block

    def __repr__(self) -> str:
        return f"EquivocatingNode({self.name!r}, B={len(self.blocklace)})"


class MalformedBlockNode:
    """Creates blocks with invalid signatures (detectable immediately)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.private_key, self.node_id = generate_keypair()

    def create_malformed_block(
        self, payload: Any, predecessors: FrozenSet[BlockId]
    ) -> Block:
        content = BlockContent(payload=payload, predecessors=predecessors)
        content_hash = sha256(content.to_bytes())
        bad_sig = sign(b"garbage", self.private_key)
        bad_id = BlockId(content_hash=content_hash, signature=bad_sig, node_id=self.node_id)
        return Block(identity=bad_id, content=content)


class ColludingNode:
    """
    Colluder — never puts ally_node_id in known_byz.
    Per §5.2: a colluder p with ally q satisfies q ∉ byz(b) for every p-block b.
    """

    def __init__(self, name: str, ally_node_id: NodeId) -> None:
        self.name = name
        self.private_key, self.node_id = generate_keypair()
        self.ally_node_id = ally_node_id
        self.blocklace: Blocklace = Blocklace()
        self.byz: FrozenSet[NodeId] = frozenset()

    def create_block(
        self,
        payload: Any,
        predecessors: FrozenSet[BlockId],
        known_byz: FrozenSet[NodeId] = frozenset(),
    ) -> Block:
        """Create block omitting ally from known_byz."""
        safe = frozenset(n for n in known_byz if n != self.ally_node_id)
        block = Block.create(payload, predecessors, self.private_key, self.node_id, safe)
        self.blocklace._blocks[block.identity] = block.content
        return block
