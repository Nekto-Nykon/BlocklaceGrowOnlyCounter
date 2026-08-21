"""
Honest P2P node — exact Python translation of TLA+ Newtry.

TLA+ variables per node:
  B[n]    -> self.blocklace       (set of accepted blocks)
  byz[n]  -> self.byz             (known Byzantine node-ids)

TLA+ actions implemented here:
  Add(n, payload)  -> self.create_block(payload)
  Receive(n)       -> self.deliver()   called by Network

Updated CanAccept matches new TLA+:

  NetworkBlocks(n) == {msg.block : msg \\in {m \\in network : m.to = n}}

  CanAccept(n, b) ==
    /\ b.parents \\subseteq SHSet(B[n])
    /\ LET ctx == B[n] \\cup {b} \\cup NetworkBlocks(n)
       IN
       \/ byz[n] \\subseteq b.known_byz
       \/ \E msg2 \\in network :
              /\ msg2.to = n
              /\ msg2.block.node \\notin byz[n]
              /\ byz[n] \\subseteq msg2.block.known_byz
              /\ Precedes(b, msg2.block, ctx)

Key changes vs previous version:
  1. ctx now includes ALL pending blocks for node n (NetworkBlocks),
     not just the single msg2.block — allows intermediate blocks
     in the network to serve as chain links for Precedes check.
  2. Added condition: msg2.block.node \\notin byz[n] — we do not use
     blocks from already-known Byzantine nodes as evidence of honesty.
  3. old_byz passed to _can_accept to match TLA+ pre-state semantics.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, Optional, Set

from .block import Block, BlockId
from .blocklace import Blocklace
from .crypto import NodeId, generate_keypair


class HonestNode:
    """Correct P2P node — TLA+ Honest node."""

    def __init__(self, name: str, valid_fn: Optional[Callable] = None) -> None:
        self.name = name
        self.private_key, self.node_id = generate_keypair()
        # B[n] in TLA+
        self.blocklace: Blocklace = Blocklace()
        # byz[n] in TLA+
        self.byz: Set[NodeId] = set()
        self.valid_fn = valid_fn

    # ── Add(n, payload) ───────────────────────────────────────────────────────

    def create_block(self, payload: Any) -> Block:
        """
        Add(n, payload) from TLA+:
          parents   = SHSet(MaxBlocks(B[n]))
          known_byz = byz[n]
        Block is added to B[n] immediately and returned for broadcast.
        """
        parents = frozenset(self.blocklace.maximal_blocks())
        block = Block.create(
            payload=payload,
            predecessors=parents,
            private_key=self.private_key,
            node_id=self.node_id,
            known_byz=frozenset(self.byz),
        )
        self.blocklace._blocks[block.identity] = block.content
        return block

    # ── Receive(n) ────────────────────────────────────────────────────────────

    def deliver(self, block: Block, pending_blocks: Dict[BlockId, Block]) -> bool:
        """
        Receive(n) from TLA+ for one message.

        Returns True  if block was accepted into B[n]   (THEN branch).
        Returns False if block was rejected              (ELSE branch).

        In both cases byz[n] is updated:
          newByz = byz[n] ∪ Eqvc(B[n] ∪ {b})

        pending_blocks: NetworkBlocks(n) — all blocks in network to n.
        Used as ctx in CanAccept disjunct B.

        TLA+ ELSE branch: B and network unchanged.
        Caller (Network) keeps message in pending when False returned.
        """
        if block.identity in self.blocklace:
            return True  # idempotent

        # Save old byz[n] — CanAccept uses pre-state value (TLA+ semantics)
        old_byz = set(self.byz)

        # newByz = byz[n] ∪ Eqvc(B[n] ∪ {b}) — always updated
        self._update_byz(block)

        if self._can_accept(block, pending_blocks, old_byz):
            self.blocklace._blocks[block.identity] = block.content
            return True
        return False

    # ── CanAccept(n, b) ───────────────────────────────────────────────────────

    def _can_accept(
        self,
        block: Block,
        pending_blocks: Dict[BlockId, Block],
        old_byz=None,
    ) -> bool:
        """
        Updated CanAccept(n, b) from TLA+:

          NetworkBlocks(n) == {msg.block : msg \\in {m \\in network : m.to = n}}

          /\ b.parents ⊆ SHSet(B[n])
          /\ LET ctx == B[n] ∪ {b} ∪ NetworkBlocks(n)
             IN
             \/ byz[n] ⊆ b.known_byz                      -- disjunct A
             \/ ∃ msg2 ∈ network :                         -- disjunct B
                    msg2.to = n
                    ∧ msg2.block.node ∉ byz[n]
                    ∧ byz[n] ⊆ msg2.block.known_byz
                    ∧ Precedes(b, msg2.block, ctx)

        ctx includes ALL pending blocks for n as intermediate chain links.
        """
        # Use pre-state byz[n] (TLA+ semantics)
        effective_byz = old_byz if old_byz is not None else self.byz

        # Closure axiom: all parents must be in B[n]
        if not all(p in self.blocklace for p in block.predecessors):
            return False

        # Disjunct A: block acknowledges all known Byzantine nodes
        if effective_byz <= set(block.known_byz):
            return True

        # Build ctx = B[n] ∪ {b} ∪ NetworkBlocks(n)
        # ctx is used for Precedes check in disjunct B
        ctx: Dict[BlockId, FrozenSet[BlockId]] = {}
        # Add all pending blocks (NetworkBlocks(n))
        for bid, blk in pending_blocks.items():
            ctx[bid] = blk.predecessors
        # Add the incoming block itself
        ctx[block.identity] = block.predecessors

        # Disjunct B: ∃ later block from same non-Byzantine author
        # that acknowledges byz[n], reachable from b in ctx
        for bid, later in pending_blocks.items():
            if bid == block.identity:
                continue
            # msg2.block.node ∉ byz[n] — not from known Byzantine
            if later.node_id in effective_byz:
                continue
            # msg2.block.node = b.node — same author (original TLA+ condition)
            if later.node_id != block.node_id:
                continue
            # byz[n] ⊆ msg2.block.known_byz
            if not (effective_byz <= set(later.known_byz)):
                continue
            # Precedes(b, msg2.block, ctx)
            if self._precedes_in_ctx(block.identity, bid, ctx):
                return True

        return False

    def _precedes_in_ctx(
        self,
        a: BlockId,
        b: BlockId,
        ctx: Dict[BlockId, FrozenSet[BlockId]],
    ) -> bool:
        """
        Precedes(a, b, ctx) where ctx = B[n] ∪ {block} ∪ NetworkBlocks(n).

        Check if a is in the causal past of b using:
          - self.blocklace (B[n])
          - ctx (pending blocks + incoming block)
        """
        visited: Set[BlockId] = set()
        # Start from b's predecessors
        stack = list(ctx.get(b, frozenset()) if b not in self.blocklace.blocks
                     else self.blocklace.blocks[b].predecessors)
        # Also include ctx predecessors for b
        if b in ctx:
            stack.extend(ctx[b])

        while stack:
            curr = stack.pop()
            if curr == a:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            # Look in blocklace first
            if curr in self.blocklace.blocks:
                stack.extend(self.blocklace.blocks[curr].predecessors)
            # Then in ctx (pending blocks)
            if curr in ctx:
                stack.extend(ctx[curr])
        return False

    # ── byz[n] update ─────────────────────────────────────────────────────────

    def _update_byz(self, incoming: Block) -> None:
        """
        newByz = byz[n] ∪ Eqvc(B[n] ∪ {b}).
        Check if incoming is concurrent with any existing block from same creator.
        """
        creator = incoming.node_id
        if creator in self.byz:
            return

        for bid in list(self.blocklace.blocks.keys()):
            if bid.node_id != creator:
                continue
            existing_precedes = self._bid_precedes_block(bid, incoming)
            incoming_precedes = self.blocklace.precedes(incoming.identity, bid)
            if not existing_precedes and not incoming_precedes:
                self.byz.add(creator)
                return

    def _bid_precedes_block(self, bid: BlockId, block: Block) -> bool:
        """Check if bid is in causal past of block (block not yet in blocklace)."""
        visited: Set[BlockId] = set()
        stack = list(block.predecessors)
        while stack:
            curr = stack.pop()
            if curr == bid:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            if curr in self.blocklace.blocks:
                stack.extend(self.blocklace.blocks[curr].predecessors)
        return False

    # ── Queries ───────────────────────────────────────────────────────────────

    def known_byzantine_nodes(self) -> Set[NodeId]:
        return self.byz | self.blocklace.equivocators()

    def polog_payloads(self) -> Dict[str, Any]:
        return {
            bid.short(): content.payload
            for bid, content in self.blocklace.polog(self.valid_fn).items()
        }

    def __repr__(self) -> str:
        return (
            f"HonestNode({self.name!r}, "
            f"B={len(self.blocklace)}, "
            f"byz={len(self.byz)})"
        )