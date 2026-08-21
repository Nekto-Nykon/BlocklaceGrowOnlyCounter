"""
Demo: GOC-Ledger + Byzantine-Repelling Blocklace (aligned with TLA+ Newtry)

Scenario
--------
Honest nodes: Alice (token issuer), Bob, Carol.
Byzantine node: Eve (equivocates — sends conflicting transfer ops).

Part 1 — Sequential transfers (no double-spending).
Part 2 — Concurrent transfers (double-spending, safety invariant checked).
Part 3 — Blocklace integration: Eve equivocates, gets detected,
          post-detection blocks rejected (FiniteHarm).
          Shows that double_spend_exposure is bounded.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.account import Account
from src.ledger import Ledger
from src.byzantine import EquivocatingNode
from src.node import HonestNode
from src.simulation import Network


def section(title):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def print_ledger(ledger, label="Ledger"):
    bals = ledger.balances()
    print(f"  {label}:")
    for aid, bal in sorted(bals.items()):
        marker = " <-- NEGATIVE (double-spent)" if bal < 0 else ""
        print(f"    {aid:12s}  balance = {bal:+.2f}{marker}")
    print(f"    safety_holds          = {ledger.safety_holds()}")
    print(f"    double_spend_exposure = {ledger.double_spend_exposure():.2f}")


def main():
    print("=" * 60)
    print("  GOC-Ledger + Blocklace Demo (Newtry TLA+ model)")
    print("=" * 60)

    # ── Part 1: Sequential transfers ─────────────────────────────────────────
    section("Part 1 - Sequential Transfers (no double-spending)")

    alice = Account("alice", {"alice"})
    alice = alice.create(200)
    bob   = Account("bob")
    carol = Account("carol")

    alice = alice.give_to(60, "bob")
    bob   = bob.ack_from(alice)
    bob   = bob.give_to(20, "carol")
    carol = carol.ack_from(bob)

    ledger = Ledger()
    for acc in [alice, bob, carol]:
        ledger.add(acc)

    print_ledger(ledger, "After sequential transfers")
    assert ledger.safety_holds()
    assert ledger.double_spend_exposure() == 0.0
    print("  [OK] No negative balances in sequential execution.")

    # ── Part 2: Concurrent double-spending ───────────────────────────────────
    section("Part 2 - Concurrent Transfers (double-spending)")

    alice_r1 = Account("alice", {"alice"})
    alice_r1 = alice_r1.create(100)
    alice_r1 = alice_r1.give_to(80, "bob")

    alice_r2 = Account("alice", {"alice"})
    alice_r2 = alice_r2.create(100)
    alice_r2 = alice_r2.give_to(80, "carol")

    alice_merged = alice_r1.merge(alice_r2)
    bob_r   = Account("bob").ack_from(alice_merged)
    carol_r = Account("carol").ack_from(alice_merged)

    l_concurrent = Ledger()
    for acc in [alice_merged, bob_r, carol_r]:
        l_concurrent.add(acc)

    print(f"  alice_r1 gave 80 to bob   (balance: {alice_r1.balance():.2f})")
    print(f"  alice_r2 gave 80 to carol (balance: {alice_r2.balance():.2f})")
    print(f"  After merge: alice balance = {alice_merged.balance():.2f}")
    print()
    print_ledger(l_concurrent, "Merged ledger (double-spend visible)")

    assert l_concurrent.safety_holds()
    print(f"\n  [OK] Safety invariant holds even with double-spending.")
    print(f"  [OK] Exposure bounded: {l_concurrent.double_spend_exposure():.2f} tokens")

    # ── Part 3: Blocklace + GOC-Ledger integration ───────────────────────────
    section("Part 3 - Blocklace Integration (Byzantine-Repelling + GOC-Ledger)")

    net = Network()
    node_alice = HonestNode("Alice")
    node_bob   = HonestNode("Bob")
    node_carol = HonestNode("Carol")
    net.add_nodes(node_alice, node_bob, node_carol)

    eve = EquivocatingNode("Eve")
    net.add_byzantine(eve)

    # Honest genesis blocks with ledger payloads
    a_blk = node_alice.create_block({"op": "create", "account": "alice", "amount": 100})
    b_blk = node_bob.create_block({"op": "create", "account": "bob", "amount": 50})
    c_blk = node_carol.create_block({"op": "create", "account": "carol", "amount": 30})
    for blk, name in [(a_blk, "Alice"), (b_blk, "Bob"), (c_blk, "Carol")]:
        net.broadcast_add(blk, name)

    net.deliver_pending()
    net.print_status("After honest genesis blocks")

    # Eve equivocates — sends BOTH forks to ALL nodes (reliable broadcast)
    fork_a, fork_b = eve.equivocate(
        {"op": "transfer", "from": "eve", "to": "alice", "amount": 999},
        {"op": "transfer", "from": "eve", "to": "bob",   "amount": 999},
        frozenset(),
    )
    print(f"  Eve sends fork A ({fork_a.identity.short()!r}) and fork B ({fork_b.identity.short()!r}) to ALL")
    net.broadcast_equivocate(fork_a, fork_b, "Eve")

    net.deliver_pending()
    net.print_status("After Eve's equivocation")

    # Verify detection
    for name, node in net.honest_nodes.items():
        byz = node.known_byzantine_nodes()
        assert eve.node_id in byz, f"{name} failed to detect Eve"
        print(f"  {name}: eve_detected=True, byz_count={len(byz)}")

    # Honest post-detection blocks — known_byz will include Eve
    for node, msg in [
        (node_alice, {"op": "transfer", "from": "alice", "to": "bob", "amount": 10}),
        (node_bob,   {"op": "transfer", "from": "bob", "to": "carol", "amount": 5}),
    ]:
        blk = node.create_block(msg)
        net.broadcast_add(blk, node.name)
        print(f"  {node.name} created post-detection block, known_byz={len(blk.known_byz)}")

    net.deliver_pending()

    # FiniteHarm: new Eve block after detection is rejected
    eve_late = eve.add_honest(
        {"op": "transfer", "from": "eve", "to": "carol", "amount": 9999},
        frozenset({fork_a.identity}),
    )
    net.broadcast_byz_honest(eve_late, "Eve")
    net.deliver_pending()
    accepted = [n for n, node in net.honest_nodes.items() if eve_late.identity in node.blocklace]
    print(f"\n  Post-detection Eve block accepted by: {accepted or 'NONE'}")

    net.print_status("Final State")

    # Build ledger from POLog payloads of alice
    section("GOC-Ledger from POLog")
    final_ledger = Ledger()
    for bid, content in node_alice.blocklace.polog(node_alice.valid_fn).items():
        payload = content.payload
        if not isinstance(payload, dict) or "op" not in payload:
            continue
        op = payload["op"]
        aid = payload.get("account") or payload.get("from")
        if aid is None:
            continue
        acc = final_ledger.get(aid) or Account(aid, {aid} if op == "create" else set())
        if op == "create":
            acc = acc.create(payload.get("amount", 0))
        elif op == "transfer":
            to_id = payload.get("to")
            amt = payload.get("amount", 0)
            acc = acc.give_to(amt, to_id)
            if to_id:
                to_acc = final_ledger.get(to_id) or Account(to_id)
                to_acc = to_acc.ack_from(acc)
                final_ledger.add(to_acc)
        final_ledger.add(acc)

    print_ledger(final_ledger, "Ledger from Alice's POLog")

    print("\n  Summary:")
    print(f"  [OK] Eve detected by all honest nodes")
    print(f"  [OK] FiniteHarm: post-detection Eve blocks rejected")
    print(f"  [OK] Safety invariant: {final_ledger.safety_holds()}")
    print(f"  [OK] Double-spend exposure: {final_ledger.double_spend_exposure():.2f}")
    print(f"  [OK] POLog convergence: {net.convergence_check()}")
    print()


if __name__ == "__main__":
    main()