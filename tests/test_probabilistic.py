"""
Probabilistic (property-based) tests using Hypothesis.
Aligned with TLA+ Newtry model and updated Python API.

Properties under test:
  1. CRDT laws for Blocklace  (commutativity, associativity, idempotency)
  2. CRDT laws for Account
  3. CRDT laws for Ledger
  4. GOC safety invariant
  5. Sequential non-negativity
  6. Double-spend exposure bounded
  7. Equivocation always detected
  8. Finite harm — post-detection blocks rejected
  9. Honest network convergence
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.account import Account
from src.block import Block
from src.blocklace import Blocklace
from src.byzantine import EquivocatingNode
from src.crypto import generate_keypair
from src.ledger import Ledger
from src.node import HonestNode
from src.simulation import Network

# ── Strategies ────────────────────────────────────────────────────────────────

amounts  = st.floats(min_value=0.01, max_value=1000.0,
                     allow_nan=False, allow_infinity=False)
payloads = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
                   min_size=1, max_size=20)

def make_block(payload, preds, sk, nid):
    return Block.create(payload, frozenset(preds), sk, nid)

def fresh_account(aid="alice", creator=True):
    return Account(aid, {aid} if creator else set())


# ─────────────────────────────────────────────────────────────────────────────
# 1. CRDT laws — Blocklace
# ─────────────────────────────────────────────────────────────────────────────

class TestBlocklaceCRDTLaws:

    @given(payloads, payloads)
    def test_merge_commutative(self, p1, p2):
        ska, nida = generate_keypair()
        skb, nidb = generate_keypair()
        bl_a, bl_b = Blocklace(), Blocklace()
        b1 = make_block(p1, [], ska, nida)
        b2 = make_block(p2, [], skb, nidb)
        bl_a._blocks[b1.identity] = b1.content
        bl_b._blocks[b2.identity] = b2.content
        ab = bl_a.merge(bl_b)
        ba = bl_b.merge(bl_a)
        assert set(ab.blocks.keys()) == set(ba.blocks.keys())

    @given(payloads, payloads, payloads)
    def test_merge_associative(self, p1, p2, p3):
        bl1, bl2, bl3 = Blocklace(), Blocklace(), Blocklace()
        for p, bl in [(p1, bl1), (p2, bl2), (p3, bl3)]:
            sk, nid = generate_keypair()
            b = make_block(p, [], sk, nid)
            bl._blocks[b.identity] = b.content
        left  = (bl1.merge(bl2)).merge(bl3)
        right = bl1.merge(bl2.merge(bl3))
        assert set(left.blocks.keys()) == set(right.blocks.keys())

    @given(payloads)
    def test_merge_idempotent(self, p):
        sk, nid = generate_keypair()
        bl = Blocklace()
        b = make_block(p, [], sk, nid)
        bl._blocks[b.identity] = b.content
        merged = bl.merge(bl)
        assert set(merged.blocks.keys()) == set(bl.blocks.keys())

    @given(payloads, payloads)
    def test_leq_after_merge(self, p1, p2):
        bl_a, bl_b = Blocklace(), Blocklace()
        for p, bl in [(p1, bl_a), (p2, bl_b)]:
            sk, nid = generate_keypair()
            b = make_block(p, [], sk, nid)
            bl._blocks[b.identity] = b.content
        merged = bl_a.merge(bl_b)
        assert all(k in merged.blocks for k in bl_a.blocks)
        assert all(k in merged.blocks for k in bl_b.blocks)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CRDT laws — Account
# ─────────────────────────────────────────────────────────────────────────────

class TestAccountCRDTLaws:

    @given(amounts, amounts)
    def test_merge_commutative(self, c1, c2):
        a = Account("x", {"x"}); a.created = c1
        b = Account("x", {"x"}); b.created = c2
        assert a.merge(b).balance() == b.merge(a).balance()

    @given(amounts, amounts, amounts)
    def test_merge_associative(self, c1, c2, c3):
        def acct(c):
            a = Account("x", {"x"}); a.created = c; return a
        a, b, c = acct(c1), acct(c2), acct(c3)
        assert (a.merge(b)).merge(c).balance() == a.merge(b.merge(c)).balance()

    @given(amounts)
    def test_merge_idempotent(self, c):
        a = Account("x", {"x"}); a.created = c
        assert a.merge(a).balance() == a.balance()

    @given(amounts, amounts)
    def test_leq_after_merge(self, c1, c2):
        a = Account("x", {"x"}); a.created = c1
        b = Account("x", {"x"}); b.created = c2
        merged = a.merge(b)
        assert a.leq(merged) and b.leq(merged)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRDT laws — Ledger
# ─────────────────────────────────────────────────────────────────────────────

class TestLedgerCRDTLaws:

    def _ledger(self, aid, created):
        a = Account(aid, {aid}); a.created = created
        l = Ledger(); l.add(a); return l

    @given(amounts, amounts)
    def test_merge_commutative(self, c1, c2):
        l1, l2 = self._ledger("alice", c1), self._ledger("alice", c2)
        assert l1.merge(l2).balances() == l2.merge(l1).balances()

    @given(amounts, amounts, amounts)
    def test_merge_associative(self, c1, c2, c3):
        l1 = self._ledger("alice", c1)
        l2 = self._ledger("alice", c2)
        l3 = self._ledger("alice", c3)
        left  = (l1.merge(l2)).merge(l3)
        right = l1.merge(l2.merge(l3))
        assert left.balances() == right.balances()

    @given(amounts)
    def test_merge_idempotent(self, c):
        l = self._ledger("alice", c)
        assert l.merge(l).balances() == l.balances()


# ─────────────────────────────────────────────────────────────────────────────
# 4-6. GOC safety invariant
# ─────────────────────────────────────────────────────────────────────────────

Op = st.one_of(
    st.tuples(st.just("create"), amounts),
    st.tuples(st.just("burn"),   amounts),
    st.tuples(st.just("give"),   amounts),
    st.just(("ack",)),
)

class TestGOCSafetyInvariant:

    @given(st.lists(Op, min_size=1, max_size=20))
    def test_safety_invariant_holds_after_any_ops(self, ops):
        alice = Account("alice", {"alice"})
        bob   = Account("bob",   {"bob"})
        for op in ops:
            if op[0] == "create": alice = alice.create(op[1])
            elif op[0] == "burn": alice = alice.burn(op[1])
            elif op[0] == "give": alice = alice.give_to(op[1], "bob")
            elif op[0] == "ack":  bob   = bob.ack_from(alice)
        l = Ledger(); l.add(alice); l.add(bob)
        assert l.safety_holds()

    @given(amounts, st.lists(amounts, min_size=1, max_size=10))
    def test_sequential_ops_never_negative(self, initial, spends):
        alice = Account("alice", {"alice"})
        alice = alice.create(initial)
        for spend in spends:
            alice = alice.give_to(spend, "bob")
            assert alice.balance() >= 0.0

    @given(amounts, st.floats(min_value=0.01, max_value=1.0,
                              allow_nan=False, allow_infinity=False))
    def test_double_spend_exposure_bounded(self, created, ratio):
        spend = created * ratio
        r1 = Account("alice", {"alice"}); r1 = r1.create(created); r1 = r1.give_to(spend, "bob")
        r2 = Account("alice", {"alice"}); r2 = r2.create(created); r2 = r2.give_to(spend, "carol")
        merged = r1.merge(r2)
        l = Ledger(); l.add(merged)
        l.add(Account("bob").ack_from(merged))
        l.add(Account("carol").ack_from(merged))
        assert l.safety_holds()
        assert l.double_spend_exposure() <= spend + 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 7. Equivocation always detected
# ─────────────────────────────────────────────────────────────────────────────

class TestEquivocationAlwaysDetected:

    @given(payloads, payloads)
    def test_any_pair_detected(self, p1, p2):
        """
        TLA+ EquivocationDetection: Eqvc(B[n]) ⊆ byz[n].
        After both blocks delivered, Byzantine node must be in byz[n].
        """
        node = HonestNode("observer")
        byz  = EquivocatingNode("byz")

        fork_a, fork_b = byz.equivocate(p1, p2, frozenset())

        # Deliver both — TLA+ Receive(n) fires twice
        # byz[n] updated on first, block accepted; byz[n] updated again on second
        node.deliver(fork_a, {fork_b.identity: fork_b})
        node.deliver(fork_b, {})

        assert byz.node_id in node.byz

    @given(payloads, payloads)
    def test_detection_order_independent(self, p1, p2):
        """Detection order does not matter — TLA+ Concurrent is symmetric."""
        node_ab = HonestNode("obs_ab")
        node_ba = HonestNode("obs_ba")
        byz = EquivocatingNode("byz")

        fork_a, fork_b = byz.equivocate(p1, p2, frozenset())

        node_ab.deliver(fork_a, {fork_b.identity: fork_b})
        node_ab.deliver(fork_b, {})

        node_ba.deliver(fork_b, {fork_a.identity: fork_a})
        node_ba.deliver(fork_a, {})

        assert byz.node_id in node_ab.byz
        assert byz.node_id in node_ba.byz


# ─────────────────────────────────────────────────────────────────────────────
# 8. Finite harm — post-detection blocks always rejected
# ─────────────────────────────────────────────────────────────────────────────

class TestFiniteHarm:

    @given(payloads, payloads, payloads)
    def test_post_detection_block_rejected(self, p1, p2, p3):
        """
        TLA+ FiniteHarm: after equivocation detected,
        further blocks from Byzantine node do not enter B[n].
        """
        node = HonestNode("observer")
        byz  = EquivocatingNode("byz")

        fork_a, fork_b = byz.equivocate(p1, p2, frozenset())
        node.deliver(fork_a, {fork_b.identity: fork_b})
        node.deliver(fork_b, {})

        assert byz.node_id in node.byz

        # Post-detection block — parent is one of the equivocating forks
        # known_byz still empty (Byzantine node never updates byz)
        post = byz.add_honest(p3, frozenset({fork_a.identity}))
        node.deliver(post, {})

        # Must NOT be in blocklace — CanAccept returns False
        assert post.identity not in node.blocklace

    @given(payloads, payloads, st.lists(payloads, min_size=1, max_size=5))
    def test_many_post_detection_blocks_all_rejected(self, p1, p2, extras):
        """Any number of post-detection blocks must all be rejected."""
        node = HonestNode("observer")
        byz  = EquivocatingNode("byz")

        fork_a, fork_b = byz.equivocate(p1, p2, frozenset())
        node.deliver(fork_a, {fork_b.identity: fork_b})
        node.deliver(fork_b, {})

        prev = frozenset({fork_a.identity})
        for p in extras:
            blk = byz.add_honest(p, prev)
            node.deliver(blk, {})
            assert blk.identity not in node.blocklace
            prev = frozenset({blk.identity})


# ─────────────────────────────────────────────────────────────────────────────
# 9. Honest network convergence — using TLA+ aligned Network
# ─────────────────────────────────────────────────────────────────────────────

class TestHonestNetworkConvergence:

    @given(st.lists(payloads, min_size=1, max_size=6))
    def test_single_node_brep(self, msgs):
        """Single honest node's blocklace is always Byzantine-repelling."""
        sk, nid = generate_keypair()
        bl = Blocklace()
        for msg in msgs:
            frontier = frozenset(bl.maximal_blocks())
            b = make_block(msg, list(frontier), sk, nid)
            bl._blocks[b.identity] = b.content
        assert bl.is_byzantine_repelling()

    @given(
        st.lists(payloads, min_size=1, max_size=4),
        st.lists(payloads, min_size=1, max_size=4),
    )
    @settings(max_examples=30)
    def test_two_honest_nodes_converge(self, msgs_a, msgs_b):
        """
        Two honest nodes broadcast blocks to each other via TLA+ aligned Network.
        After deliver_pending() they must have identical B[n] sets.
        """
        net   = Network()
        alice = HonestNode("alice")
        bob   = HonestNode("bob")
        net.add_nodes(alice, bob)

        for msg in msgs_a:
            blk = alice.create_block(msg)
            net.broadcast_add(blk, "alice")

        for msg in msgs_b:
            blk = bob.create_block(msg)
            net.broadcast_add(blk, "bob")

        net.deliver_pending()

        alice_keys = set(alice.blocklace.blocks.keys())
        bob_keys   = set(bob.blocklace.blocks.keys())
        assert alice_keys == bob_keys
        assert alice.blocklace.is_byzantine_repelling()
        assert bob.blocklace.is_byzantine_repelling()

    @given(st.lists(payloads, min_size=1, max_size=4))
    def test_polog_subset_of_blocklace(self, msgs):
        """polog is always a subset of the full blocklace."""
        sk, nid = generate_keypair()
        bl = Blocklace()
        for msg in msgs:
            frontier = frozenset(bl.maximal_blocks())
            b = make_block(msg, list(frontier), sk, nid)
            bl._blocks[b.identity] = b.content
        assert set(bl.polog().keys()) <= set(bl.blocks.keys())

    @given(
        st.lists(payloads, min_size=1, max_size=3),
        st.lists(payloads, min_size=1, max_size=3),
        payloads, payloads,
    )
    @settings(max_examples=20)
    def test_convergence_after_equivocation(self, msgs_a, msgs_b, ep1, ep2):
        """
        After Byzantine equivocation and deliver_pending(),
        all honest nodes detect Byzantine node and have identical POLogs.
        Models TLA+ Fairness: WF_vars(Receive(n)) for all n.
        """
        net   = Network()
        alice = HonestNode("alice")
        bob   = HonestNode("bob")
        net.add_nodes(alice, bob)

        eve = EquivocatingNode("eve")
        net.add_byzantine(eve)

        # Honest blocks
        for msg in msgs_a:
            blk = alice.create_block(msg)
            net.broadcast_add(blk, "alice")
        for msg in msgs_b:
            blk = bob.create_block(msg)
            net.broadcast_add(blk, "bob")

        # Equivocation — both blocks to all nodes (TLA+ ByzantineEquivocate)
        fork_a, fork_b = eve.equivocate(ep1, ep2, frozenset())
        net.broadcast_equivocate(fork_a, fork_b, "eve")

        net.deliver_pending()

        # Both nodes must detect Eve
        assert eve.node_id in alice.byz
        assert eve.node_id in bob.byz

        # POLogs must converge
        assert net.convergence_check()

        # TLA+ POLogSafety + ByzRepelling:
        # Once equivocator detected, no equivocating PAIR should both be in polog.
        # Exactly one fork may be accepted as first evidence (CanAccept disjunct B),
        # but the equivocating pair cannot both be present in polog.
        alice_polog = set(alice.blocklace.polog().keys())
        bob_polog   = set(bob.blocklace.polog().keys())

        # Both forks cannot be in the same polog simultaneously
        alice_has_both = (fork_a.identity in alice_polog and
                          fork_b.identity in alice_polog)
        bob_has_both   = (fork_a.identity in bob_polog and
                          fork_b.identity in bob_polog)
        assert not alice_has_both, "Alice polog has both equivocating blocks"
        assert not bob_has_both,   "Bob polog has both equivocating blocks"