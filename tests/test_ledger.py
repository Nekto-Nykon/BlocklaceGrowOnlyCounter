"""
Unit tests for the GOC Account and GOC Ledger CRDTs.

Covers:
  - Account create / burn / give / ack operations
  - CRDT merge (element-wise max)
  - Partial order (leq)
  - Balance computation
  - Ledger convergence and safety invariant
  - Double-spend exposure
"""
import pytest

from src.account import Account
from src.ledger import Ledger


# ── Account basics ────────────────────────────────────────────────────────────

class TestAccountOperations:
    def test_initial_balance_zero(self):
        a = Account("alice", {"alice"})
        assert a.balance() == 0.0

    def test_create_increases_balance(self):
        a = Account("alice", {"alice"})
        a2 = a.create(100)
        assert a2.balance() == 100.0
        assert a.balance() == 0.0   # original unchanged

    def test_create_rejects_non_creator(self):
        a = Account("alice", {"bob"})  # alice is not a creator
        a2 = a.create(100)
        assert a2.balance() == 0.0

    def test_burn_reduces_balance(self):
        a = Account("alice", {"alice"})
        a = a.create(100)
        a = a.burn(40)
        assert a.balance() == 60.0

    def test_burn_rejected_if_insufficient(self):
        a = Account("alice", {"alice"})
        a = a.create(50)
        a2 = a.burn(100)
        assert a2.balance() == 50.0  # no change

    def test_give_reduces_balance(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        alice = alice.give_to(30, "bob")
        assert alice.balance() == 70.0

    def test_give_rejected_if_insufficient(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(20)
        alice2 = alice.give_to(50, "bob")
        assert alice2.balance() == 20.0

    def test_ack_increases_balance(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        alice = alice.give_to(40, "bob")

        bob = Account("bob")
        bob = bob.ack_from(alice)
        assert bob.balance() == 40.0

    def test_double_ack_idempotent(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        alice = alice.give_to(40, "bob")

        bob = Account("bob")
        bob = bob.ack_from(alice)
        bob2 = bob.ack_from(alice)
        assert bob2.balance() == 40.0  # same as after first ack

    def test_unacked_from(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        alice = alice.give_to(40, "bob")
        bob = Account("bob")
        assert bob.unacked_from(alice) == 40.0
        bob = bob.ack_from(alice)
        assert bob.unacked_from(alice) == 0.0


# ── Account CRDT ─────────────────────────────────────────────────────────────

class TestAccountCRDT:
    def test_merge_commutativity(self):
        a = Account("x", {"x"})
        a = a.create(100)
        b = Account("x", {"x"})
        b.created = 80
        b.burned = 10
        assert a.merge(b).balance() == a.merge(b).balance()
        assert b.merge(a).balance() == a.merge(b).balance()

    def test_merge_idempotent(self):
        a = Account("x", {"x"})
        a = a.create(50)
        assert a.merge(a).balance() == a.balance()

    def test_merge_takes_max(self):
        a = Account("x", {"x"})
        a.created = 50
        b = Account("x", {"x"})
        b.created = 100
        merged = a.merge(b)
        assert merged.created == 100

    def test_merge_different_ids_raises(self):
        a = Account("alice")
        b = Account("bob")
        with pytest.raises(ValueError):
            a.merge(b)

    def test_leq_reflexive(self):
        a = Account("x", {"x"})
        a = a.create(100)
        assert a.leq(a)

    def test_leq_after_merge(self):
        a = Account("x", {"x"})
        a = a.create(100)
        b = Account("x", {"x"})
        b.created = 200
        merged = a.merge(b)
        assert a.leq(merged)
        assert b.leq(merged)


# ── Ledger ────────────────────────────────────────────────────────────────────

class TestLedger:
    def _setup_ledger(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        ledger = Ledger()
        ledger.add(alice)
        return ledger, alice

    def test_add_account(self):
        ledger, alice = self._setup_ledger()
        assert "alice" in ledger.account_ids()
        assert ledger.get("alice").balance() == 100.0

    def test_merge_ledgers(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        bob = Account("bob", {"bob"})
        bob = bob.create(50)
        l1 = Ledger()
        l1.add(alice)
        l2 = Ledger()
        l2.add(bob)
        merged = l1.merge(l2)
        assert merged.get("alice").balance() == 100.0
        assert merged.get("bob").balance() == 50.0

    def test_convergence(self):
        """Two replicas that receive operations in different orders must converge."""
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        alice2 = alice.give_to(30, "bob")

        bob = Account("bob")
        bob2 = bob.ack_from(alice2)

        l1 = Ledger()
        l1.add(alice2)
        l1.add(bob2)

        l2 = Ledger()
        l2.add(bob2)
        l2.add(alice2)

        merged = l1.merge(l2)
        assert merged.get("alice").balance() == 70.0
        assert merged.get("bob").balance() == 30.0

    def test_safety_holds_simple(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(200)
        ledger = Ledger()
        ledger.add(alice)
        assert ledger.safety_holds()

    def test_safety_holds_after_transfer(self):
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        alice = alice.give_to(40, "bob")
        bob = Account("bob")
        bob = bob.ack_from(alice)

        ledger = Ledger()
        ledger.add(alice)
        ledger.add(bob)
        assert ledger.safety_holds()

    def test_double_spend_exposure(self):
        """
        Simulate concurrent spend: two replicas each observe one spend,
        then merge.  The merged ledger may have a negative balance.
        The safety invariant still holds.
        """
        # Replica 1: alice sends 70 to bob
        alice_r1 = Account("alice", {"alice"})
        alice_r1 = alice_r1.create(100)
        alice_r1 = alice_r1.give_to(70, "bob")

        # Replica 2: alice concurrently sends 70 to carol
        alice_r2 = Account("alice", {"alice"})
        alice_r2 = alice_r2.create(100)
        alice_r2 = alice_r2.give_to(70, "carol")

        merged_alice = alice_r1.merge(alice_r2)

        l = Ledger()
        l.add(merged_alice)
        l.add(Account("bob").ack_from(merged_alice))
        l.add(Account("carol").ack_from(merged_alice))

        # alice.balance = 100 - 70 - 70 = -40
        assert merged_alice.balance() == -40.0
        assert l.double_spend_exposure() == 40.0
        assert l.safety_holds()

    def test_no_negative_without_concurrency(self):
        """Sequential operations cannot produce a negative balance."""
        alice = Account("alice", {"alice"})
        alice = alice.create(100)
        alice = alice.give_to(60, "bob")
        alice = alice.give_to(30, "carol")
        # would exceed balance — should be rejected
        alice2 = alice.give_to(20, "dave")
        assert alice2.balance() == alice.balance()  # no change
        assert alice.balance() == 10.0
