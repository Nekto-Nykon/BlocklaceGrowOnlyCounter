"""
Network simulation — exact translation of TLA+ Newtry.

TLA+ network model:
  - network is a SET of messages {from, to, block}
  - Add(n, payload) adds messages to ALL Nodes \\ {n}
  - ByzantineEquivocate adds both blocks to ALL Nodes \\ {bz}
  - ByzantineAddHonest adds one block to ALL Honest nodes
  - Receive(n) processes ONE message from network addressed to n:
      THEN: accepts block, removes message from network
      ELSE: updates byz[n] only, message STAYS in network
  - Fairness: WF_vars(Receive(n)) for all n — every pending
    message is eventually processed

No Gossip action in TLA+. Dissemination comes from:
  1. Add/ByzantineEquivocate broadcast to all nodes immediately
  2. Messages stay in network until CanAccept becomes True
  3. deliver_pending() retries all pending messages (models fairness)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .block import Block, BlockId
from .byzantine import EquivocatingNode
from .node import HonestNode


@dataclass
class Message:
    """One network message: {from, to, block} as in TLA+."""
    sender: str
    recipient: str
    block: Block


class Network:
    """
    Simulated P2P network with persistent messages.

    network (TLA+) -> self._pending: list of undelivered messages
    Honest nodes   -> self.honest_nodes
    Byzantine nodes-> self.byz_nodes (tracked for ByzantineAddHonest targeting)
    """

    def __init__(self) -> None:
        self.honest_nodes: Dict[str, HonestNode] = {}
        self.byz_nodes: Dict[str, EquivocatingNode] = {}
        # Pending messages — messages stay here until accepted (TLA+ ELSE branch)
        self._pending: List[Message] = []

    # ── Node management ───────────────────────────────────────────────────────

    def add_node(self, node: HonestNode) -> None:
        self.honest_nodes[node.name] = node

    def add_nodes(self, *nodes: HonestNode) -> None:
        for n in nodes:
            self.add_node(n)

    def add_byzantine(self, node: EquivocatingNode) -> None:
        """Register Byzantine node (needed for ByzantineAddHonest targeting)."""
        self.byz_nodes[node.name] = node

    @property
    def all_node_names(self) -> Set[str]:
        return set(self.honest_nodes) | set(self.byz_nodes)

    # ── Add(n, payload) broadcast ─────────────────────────────────────────────

    def broadcast_add(self, block: Block, sender_name: str) -> None:
        """
        Add(n, payload) in TLA+:
          network' = network ∪ {[from->n, to->m, block->b] : m ∈ Nodes \\ {n}}
        Sends to ALL nodes (honest and Byzantine) except sender.
        """
        for name in self.all_node_names:
            if name != sender_name:
                self._pending.append(Message(sender_name, name, block))

    # ── ByzantineEquivocate broadcast ─────────────────────────────────────────

    def broadcast_equivocate(
        self,
        block_a: Block,
        block_b: Block,
        byz_name: str,
    ) -> None:
        """
        ByzantineEquivocate(bz) in TLA+:
          network' = network
            ∪ {blockA -> m : m ∈ Nodes \\ {bz}}
            ∪ {blockB -> m : m ∈ Nodes \\ {bz}}
        Sends BOTH blocks to ALL nodes except the Byzantine sender.
        """
        for name in self.all_node_names:
            if name != byz_name:
                self._pending.append(Message(byz_name, name, block_a))
                self._pending.append(Message(byz_name, name, block_b))

    # ── ByzantineAddHonest broadcast ──────────────────────────────────────────

    def broadcast_byz_honest(self, block: Block, byz_name: str) -> None:
        """
        ByzantineAddHonest(bz, payload) in TLA+:
          network' = network ∪ {block -> m : m ∈ Honest}
        Sends only to Honest nodes.
        """
        for name in self.honest_nodes:
            self._pending.append(Message(byz_name, name, block))

    # ── Receive(n) — one step ─────────────────────────────────────────────────

    def _pending_for(self, node_name: str) -> Dict[BlockId, Block]:
        """All blocks in pending messages addressed to node_name."""
        return {
            msg.block.identity: msg.block
            for msg in self._pending
            if msg.recipient == node_name
        }

    def deliver_one(self, node_name: str) -> bool:
        """
        Try to deliver one pending message to node_name.
        TLA+ Receive(n): pick one msg ∈ network with msg.to = n.
          THEN: accept, remove from network → return True
          ELSE: update byz only, message stays → return False

        Returns True if any message was accepted, False otherwise.
        """
        node = self.honest_nodes.get(node_name)
        if node is None:
            return False

        pending = self._pending_for(node_name)

        for i, msg in enumerate(self._pending):
            if msg.recipient != node_name:
                continue
            accepted = node.deliver(msg.block, pending)
            if accepted:
                self._pending.pop(i)
                return True
            # ELSE: byz[n] updated inside deliver(), message stays in _pending
        return False

    # ── deliver_pending() — models WF fairness ────────────────────────────────

    def deliver_pending(self, max_rounds: int = 50) -> int:
        """
        Repeatedly call Receive(n) for all honest nodes until no progress.
        Models the Weak Fairness condition:
          WF_vars(Receive(n)) for all n ∈ Nodes

        A 'round' is one full pass over all honest nodes.
        Returns number of rounds until stable.
        """
        for r in range(max_rounds):
            progress = False
            for name in list(self.honest_nodes.keys()):
                # Keep trying this node until no more messages are accepted
                while self.deliver_one(name):
                    progress = True
            if not progress:
                return r + 1
        return max_rounds

    # ── Status reporting ──────────────────────────────────────────────────────

    def print_status(self, title: str = "Network Status") -> None:
        pending_counts = {}
        for msg in self._pending:
            pending_counts[msg.recipient] = pending_counts.get(msg.recipient, 0) + 1

        print(f"\n{'=' * 58}")
        print(f"  {title}")
        print(f"{'=' * 58}")
        print(f"  Total pending messages: {len(self._pending)}")
        for name, node in self.honest_nodes.items():
            byz = node.known_byzantine_nodes()
            polog = node.polog_payloads()
            pend = pending_counts.get(name, 0)
            print(
                f"  {name:<12} "
                f"B={len(node.blocklace):3d}  "
                f"byz={len(node.byz):2d}  "
                f"polog={len(polog):3d}  "
                f"pending={pend:2d}"
            )
        print(f"{'=' * 58}\n")

    def convergence_check(self) -> bool:
        """True if all honest nodes have identical PO-Log block sets."""
        pologs = [frozenset(n.polog_payloads().keys()) for n in self.honest_nodes.values()]
        return len(pologs) == 0 or all(p == pologs[0] for p in pologs)
