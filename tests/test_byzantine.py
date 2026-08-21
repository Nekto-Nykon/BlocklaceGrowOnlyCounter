"""
Integration tests for Byzantine-repelling behaviour.
Updated to use TLA+ aligned API.
"""
import pytest

from src.block import Block
from src.blocklace import Blocklace
from src.byzantine import ColludingNode, EquivocatingNode, MalformedBlockNode
from src.node import HonestNode
from src.simulation import Network


def build_network(*names):
    net = Network()
    nodes = {}
    for n in names:
        h = HonestNode(n)
        net.add_node(h)
        nodes[n] = h
    return net, nodes


# ── Equivocation ──────────────────────────────────────────────────────────────

class TestEquivocation:

    def test_equivocator_detected_by_single_node(self):
        """TLA+: EquivocationDetection — Eqvc(B[n]) ⊆ byz[n]."""
        alice = HonestNode("alice")
        byz = EquivocatingNode("byz")

        fork_a, fork_b = byz.equivocate("fork_a", "fork_b", frozenset())
        # TLA+ Receive(alice): deliver both blocks
        alice.deliver(fork_a, {fork_b.identity: fork_b})
        alice.deliver(fork_b, {})

        assert byz.node_id in alice.byz

    def test_equivocator_accepted_as_evidence_in_polog(self):
        """
        TLA+ CanAccept: first equivocating block accepted as evidence.
        Second rejected (FiniteHarm). So only one fork enters blocklace.
        With only one fork in blocklace, EquivBlocks is empty → fork IS in polog.
        But equivocator is still detected via byz[n] update.
        """
        alice = HonestNode("alice")
        byz = EquivocatingNode("byz")

        fork_a, fork_b = byz.equivocate("fork_a", "fork_b", frozenset())
        alice.deliver(fork_a, {fork_b.identity: fork_b})

        # First fork accepted as evidence
        assert fork_a.identity in alice.blocklace

        alice.deliver(fork_b, {})

        # Equivocator detected via byz[n]
        assert byz.node_id in alice.byz
        # Both forks accepted in blocklace (TLA+: byz[n] = {} when CanAccept runs)
        assert fork_a.identity in alice.blocklace
        assert fork_b.identity in alice.blocklace
        # Both excluded from POLog (TLA+: b ∉ EquivBlocks(b.node, bl))
        polog = alice.blocklace.polog()
        assert fork_a.identity not in polog
        assert fork_b.identity not in polog

    def test_equivocator_detected_across_network(self):
        """TLA+ ByzantineEquivocate sends both to all Nodes."""
        net, nodes = build_network("alice", "bob", "carol")
        byz = EquivocatingNode("byz")
        net.add_byzantine(byz)

        fork_a, fork_b = byz.equivocate("fork_a", "fork_b", frozenset())
        net.broadcast_equivocate(fork_a, fork_b, "byz")
        net.deliver_pending()

        for name, node in nodes.items():
            assert byz.node_id in node.byz, f"{name} did not detect equivocator"

    def test_honest_blocks_still_in_polog_after_equivocator(self):
        """Honest blocks remain in polog after equivocator detected."""
        alice = HonestNode("alice")
        byz = EquivocatingNode("byz")

        honest_blk = alice.create_block("honest_payload")
        fork_a, fork_b = byz.equivocate("evil_a", "evil_b", frozenset())
        alice.deliver(fork_a, {fork_b.identity: fork_b})
        alice.deliver(fork_b, {})

        payloads = list(alice.polog_payloads().values())
        assert "honest_payload" in payloads
        assert byz.node_id in alice.byz


# ── Malformed blocks ──────────────────────────────────────────────────────────

class TestMalformedBlocks:

    def test_malformed_creator_marked_byzantine(self):
        """Force-insert malformed block — creator detected via byzantine_nodes()."""
        alice = HonestNode("alice")
        byz = MalformedBlockNode("byz")

        bad = byz.create_malformed_block("sneaky", frozenset())
        alice.blocklace._blocks[bad.identity] = bad.content

        assert byz.node_id in alice.blocklace.byzantine_nodes()

    def test_malformed_not_in_polog(self):
        """Malformed block excluded from polog."""
        alice = HonestNode("alice")
        byz = MalformedBlockNode("byz")

        bad = byz.create_malformed_block("sneaky", frozenset())
        alice.blocklace._blocks[bad.identity] = bad.content

        assert bad.identity not in alice.blocklace.polog()


# ── Colluding nodes ───────────────────────────────────────────────────────────

class TestColludingNodes:

    def test_colluder_block_rejected_after_detection(self):
        """
        After equivocator detected, colluder's block without known_byz acknowledgement
        fails CanAccept disjunct A (byz[n] ⊄ b.known_byz).
        Colluder's block stays out of blocklace.
        """
        alice = HonestNode("alice")
        byz_eqv = EquivocatingNode("byz_eqv")
        colluder = ColludingNode("colluder", byz_eqv.node_id)

        fork_a, fork_b = byz_eqv.equivocate("eq_a", "eq_b", frozenset())
        alice.deliver(fork_a, {fork_b.identity: fork_b})
        alice.deliver(fork_b, {})

        assert byz_eqv.node_id in alice.byz

        # Colluder creates block with empty known_byz (omits byz_eqv)
        c_block = colluder.create_block("collude_msg", frozenset())
        alice.deliver(c_block, {})

        # Block rejected: byz[alice]={byz_eqv} but c_block.known_byz={}
        assert c_block.identity not in alice.blocklace


# ── Full network convergence ──────────────────────────────────────────────────

class TestNetworkConvergence:

    def test_honest_network_converges(self):
        """TLA+: after deliver_pending, all honest nodes have same polog."""
        net, nodes = build_network("alice", "bob", "carol")

        for name, node in nodes.items():
            blk = node.create_block(f"msg_from_{name}")
            net.broadcast_add(blk, name)

        net.deliver_pending()

        pologs = [set(n.polog_payloads().values()) for n in nodes.values()]
        assert all(p == pologs[0] for p in pologs)

    def test_network_with_equivocator_all_nodes_detect_byzantine(self):
        """
        TLA+ ByzantineEquivocate sends both forks to all Nodes.
        After deliver_pending: every honest node detects equivocator,
        honest blocks are in polog, Eve's forks are NOT in polog.
        """
        net, nodes = build_network("alice", "bob")
        byz = EquivocatingNode("byz")
        net.add_byzantine(byz)

        ab = nodes["alice"].create_block("alice_msg")
        net.broadcast_add(ab, "alice")
        bb = nodes["bob"].create_block("bob_msg")
        net.broadcast_add(bb, "bob")

        fork_a, fork_b = byz.equivocate("evil_1", "evil_2", frozenset())
        net.broadcast_equivocate(fork_a, fork_b, "byz")

        net.deliver_pending()

        for name, node in nodes.items():
            payloads = set(node.polog_payloads().values())
            assert "alice_msg" in payloads, f"{name} missing alice_msg"
            assert "bob_msg" in payloads, f"{name} missing bob_msg"
            assert byz.node_id in node.byz, f"{name} did not detect equivocator"
            # TLA+ FiniteHarm: both forks cannot both be in polog
            # (only one fork accepted as evidence, second rejected)
            polog = node.blocklace.polog()
            both_in_polog = (fork_a.identity in polog and fork_b.identity in polog)
            assert not both_in_polog, f"{name} has both equivocating forks in polog"

        # Post-detection block rejected (TLA+: FiniteHarm)
        e3 = byz.add_honest("evil_3_post", frozenset({fork_a.identity}))
        for name, node in nodes.items():
            node.deliver(e3, {})
        for name, node in nodes.items():
            assert e3.identity not in node.blocklace, \
                f"{name} incorrectly accepted post-detection block"