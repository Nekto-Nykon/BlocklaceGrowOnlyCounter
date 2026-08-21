"""
Blocklace CRDT — §2–5 of the Blocklace paper.
Updated polog() to match TLA+ POLog exactly.

TLA+ POLog(bl):
  {b ∈ bl :
    \\/ b = Genesis
    \\/ /\\ WF(b, DownwardClosure(b, bl))
       /\\ b.node ∉ Eqvc(DownwardClosure(b, bl))
       /\\ b ∉ EquivBlocks(b.node, bl)}

Key: EquivBlocks uses the GLOBAL bl, not just DownwardClosure.
This means an equivocating block is excluded from polog as soon as
its equivocating pair is visible in the full blocklace.
"""
from __future__ import annotations

from typing import Callable, Dict, FrozenSet, Optional, Set

from .block import Block, BlockContent, BlockId
from .crypto import NodeId


class Blocklace:

    def __init__(self, blocks: Optional[Dict[BlockId, BlockContent]] = None) -> None:
        self._blocks: Dict[BlockId, BlockContent] = dict(blocks) if blocks else {}

    @property
    def blocks(self) -> Dict[BlockId, BlockContent]:
        return self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, bid: BlockId) -> bool:
        return bid in self._blocks

    def get_block(self, bid: BlockId) -> Optional[Block]:
        if bid not in self._blocks:
            return None
        return Block(identity=bid, content=self._blocks[bid])

    def add_block(self, block: Block) -> bool:
        if not all(p in self._blocks for p in block.predecessors):
            return False
        self._blocks[block.identity] = block.content
        return True

    def merge(self, other: "Blocklace") -> "Blocklace":
        merged = dict(self._blocks)
        merged.update(other._blocks)
        return Blocklace(merged)

    # ── Structural queries ────────────────────────────────────────────────────

    def maximal_blocks(self) -> Set[BlockId]:
        """max_≺(B): blocks not pointed to by any other block."""
        pointed: Set[BlockId] = set()
        for content in self._blocks.values():
            pointed.update(content.predecessors)
        return set(self._blocks.keys()) - pointed

    # ── Causal order ──────────────────────────────────────────────────────────

    def precedes(self, a: BlockId, b: BlockId) -> bool:
        """a ≺ b: a reachable from b following predecessor pointers."""
        if a not in self._blocks or b not in self._blocks:
            return False
        visited: Set[BlockId] = set()
        stack = list(self._blocks[b].predecessors)
        while stack:
            curr = stack.pop()
            if curr == a:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            if curr in self._blocks:
                stack.extend(self._blocks[curr].predecessors)
        return False

    def concurrent(self, a: BlockId, b: BlockId) -> bool:
        return a != b and not self.precedes(a, b) and not self.precedes(b, a)

    # ── Downward closure ──────────────────────────────────────────────────────

    def closure(self, bid: BlockId) -> Set[BlockId]:
        """≤b: downward closure of bid (causal past + itself)."""
        result: Set[BlockId] = set()
        stack = [bid]
        while stack:
            curr = stack.pop()
            if curr in result:
                continue
            result.add(curr)
            if curr in self._blocks:
                stack.extend(self._blocks[curr].predecessors)
        return result

    def strict_past(self, bid: BlockId) -> Set[BlockId]:
        """≺b: strict causal past (closure minus bid)."""
        return self.closure(bid) - {bid}

    def sub_blocklace(self, bids: Set[BlockId]) -> "Blocklace":
        return Blocklace({b: self._blocks[b] for b in bids if b in self._blocks})

    # ── Well-formedness §4.1 ──────────────────────────────────────────────────

    def is_well_formed(self, bid: BlockId) -> bool:
        """
        WF(b, bl): predecessors form an antichain (no two comparable).
        Mirrors TLA+ WF(b, DownwardClosure(b, bl)).
        """
        if bid not in self._blocks:
            return False
        block = self.get_block(bid)
        if not block.has_valid_signature():
            return False
        preds = list(self._blocks[bid].predecessors)
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if self.precedes(preds[i], preds[j]) or self.precedes(preds[j], preds[i]):
                    return False
        return True

    # ── Equivocation detection §4.2 ───────────────────────────────────────────

    def equivocators(self) -> Set[NodeId]:
        """
        Eqvc(bl): nodes with two concurrent blocks in bl.
        TLA+: {p ∈ Nodes : ∃ a,b ∈ bl : a≠b ∧ a.node=p=b.node ∧ Concurrent(a,b,bl)}
        """
        node_blocks: Dict[NodeId, list] = {}
        for bid in self._blocks:
            node_blocks.setdefault(bid.node_id, []).append(bid)

        result: Set[NodeId] = set()
        for nid, bids in node_blocks.items():
            for i in range(len(bids)):
                for j in range(i + 1, len(bids)):
                    if self.concurrent(bids[i], bids[j]):
                        result.add(nid)
                        break
                if nid in result:
                    break
        return result

    def equiv_blocks(self, node_id: NodeId) -> Set[BlockId]:
        """
        EquivBlocks(p, bl) in TLA+:
        {b ∈ bl : b.node=p ∧ ∃ b2 ∈ bl : b2.node=p ∧ b2≠b ∧ Concurrent(b,b2,bl)}
        """
        node_bids = [bid for bid in self._blocks if bid.node_id == node_id]
        result: Set[BlockId] = set()
        for i in range(len(node_bids)):
            for j in range(i + 1, len(node_bids)):
                a, b = node_bids[i], node_bids[j]
                if self.concurrent(a, b):
                    result.add(a)
                    result.add(b)
        return result

    def _malformed_or_invalid_creators(
        self, valid_fn: Optional[Callable] = None
    ) -> Set[NodeId]:
        result: Set[NodeId] = set()
        for bid in self._blocks:
            if not self.is_well_formed(bid):
                result.add(bid.node_id)
                continue
            if valid_fn is not None:
                past_bl = self.sub_blocklace(self.strict_past(bid))
                if not valid_fn(self._blocks[bid].payload, past_bl):
                    result.add(bid.node_id)
        return result

    def byzantine_nodes(self, valid_fn: Optional[Callable] = None) -> Set[NodeId]:
        """byz(B) = Eqvc(B) ∪ {creators of malformed/invalid blocks}."""
        return self.equivocators() | self._malformed_or_invalid_creators(valid_fn)

    def byzantine_nodes_in_closure(
        self, bid: BlockId, valid_fn: Optional[Callable] = None
    ) -> Set[NodeId]:
        """byz(≺b): Byzantine nodes in strict causal past of bid."""
        return self.sub_blocklace(self.strict_past(bid)).byzantine_nodes(valid_fn)

    # ── PO-Log §4.3 — exact TLA+ POLog ───────────────────────────────────────

    def polog(self, valid_fn: Optional[Callable] = None) -> Dict[BlockId, BlockContent]:
        """
        POLog(bl) from TLA+:
          {b ∈ bl :
            \\/ b = Genesis
            \\/ /\\ WF(b, DownwardClosure(b, bl))
               /\\ b.node ∉ Eqvc(DownwardClosure(b, bl))
               /\\ b ∉ EquivBlocks(b.node, bl)}

        Note: EquivBlocks uses the GLOBAL bl (not just downward closure).
        This excludes equivocating blocks as soon as their pair is visible.
        """
        result: Dict[BlockId, BlockContent] = {}
        for bid, content in self._blocks.items():
            block = self.get_block(bid)

            # Genesis always included
            if (bid.node_id.hex() == "genesis" or
                    content.payload == "bottom" and not content.predecessors):
                result[bid] = content
                continue

            # WF(b, DownwardClosure(b, bl))
            past_bl = self.sub_blocklace(self.closure(bid))
            if not past_bl.is_well_formed(bid):
                continue

            # b.node ∉ Eqvc(DownwardClosure(b, bl))
            if bid.node_id in past_bl.equivocators():
                continue

            # b ∉ EquivBlocks(b.node, bl)  — uses GLOBAL self
            if bid in self.equiv_blocks(bid.node_id):
                continue

            # valid_fn check
            if valid_fn is not None:
                strict_past_bl = self.sub_blocklace(self.strict_past(bid))
                if not valid_fn(content.payload, strict_past_bl):
                    continue

            result[bid] = content
        return result

    # ── Byzantine-repelling check §5.2 ───────────────────────────────────────

    def is_byzantine_repelling(self, valid_fn: Optional[Callable] = None) -> bool:
        """brep(B): greedy peel check per Definition 5.3."""
        remaining = Blocklace(dict(self._blocks))
        while remaining._blocks:
            peeled = False
            for bid in list(remaining.maximal_blocks()):
                prefix = Blocklace({b: c for b, c in remaining._blocks.items() if b != bid})
                byz_full   = remaining.byzantine_nodes(valid_fn)
                byz_prefix = prefix.byzantine_nodes(valid_fn)

                # Condition 1: this block reveals a new Byzantine node
                if byz_prefix < byz_full:
                    remaining = prefix
                    peeled = True
                    break

                # Condition 2: honest creator AND causal past knew all Byzantines
                if bid.node_id not in byz_full:
                    past_byz = remaining.sub_blocklace(
                        remaining.strict_past(bid)
                    ).byzantine_nodes(valid_fn)
                    if byz_prefix <= past_byz:
                        remaining = prefix
                        peeled = True
                        break

            if not peeled:
                return False
        return True