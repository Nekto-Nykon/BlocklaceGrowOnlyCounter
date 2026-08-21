"""
Demo: Blocklace + GOC-Ledger — exact TLA+ Newtry alignment.

Network model:
  - No Gossip. Add() broadcasts to ALL nodes immediately.
  - Messages stay in network until CanAccept becomes True (persistent network).
  - deliver_pending() models WF fairness (Receive eventually fires for all).

Scenario:
  Alice, Bob, Carol — honest nodes.
  Eve               — Byzantine equivocating node.

  Round 1: Token creation (honest genesis blocks)
  Round 2: Sequential transfer Alice -> Bob
  Round 3: Eve equivocates (double-spend)
  Round 4: Honest post-detection blocks (known_byz = {Eve})
  Final:   Verify TLA+ invariants + GOC-Ledger safety
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.account import Account
from src.byzantine import EquivocatingNode
from src.ledger import Ledger
from src.node import HonestNode
from src.simulation import Network


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def print_ledger(ledger: Ledger, label: str = "Ledger") -> None:
    bals = ledger.balances()
    print(f"  {label}:")
    for aid, bal in sorted(bals.items()):
        marker = "  <-- NEGATIVE" if bal < 0 else ""
        print(f"    {aid:12s}  balance = {bal:+.2f}{marker}")
    print(f"    safety_holds          = {ledger.safety_holds()}")
    print(f"    double_spend_exposure = {ledger.double_spend_exposure():.2f}")


def build_ledger_from_polog(node: HonestNode) -> Ledger:
    """Reconstruct GOC-Ledger from node's POLog."""
    accounts: dict = {}
    polog = node.blocklace.polog()
    ops = sorted(
        [(bid, content.payload) for bid, content in polog.items()
         if isinstance(content.payload, dict) and
         content.payload.get("type") in ("create", "transfer", "ack")],
        key=lambda x: x[0].content_hash,
    )
    for bid, op in ops:
        t = op["type"]
        if t == "create":
            aid = op["account"]
            if aid not in accounts:
                accounts[aid] = Account(aid, {aid})
            accounts[aid] = accounts[aid].create(op["amount"])
        elif t == "transfer":
            frm, to = op["from"], op["to"]
            if frm not in accounts:
                accounts[frm] = Account(frm, {frm})
            accounts[frm] = accounts[frm].give_to(op["amount"], to)
            if to not in accounts:
                accounts[to] = Account(to)
        elif t == "ack":
            frm, to = op["from"], op["to"]
            if to not in accounts:
                accounts[to] = Account(to)
            if frm in accounts:
                accounts[to] = accounts[to].ack_from(accounts[frm])
    ledger = Ledger()
    for acc in accounts.values():
        ledger.add(acc)
    return ledger


def main() -> None:
    print("=" * 60)
    print("  Blocklace + GOC-Ledger Demo  (TLA+ Newtry aligned)")
    print("=" * 60)

    # ── Setup ─────────────────────────────────────────────────────────────────
    net = Network()
    alice = HonestNode("Alice")
    bob   = HonestNode("Bob")
    carol = HonestNode("Carol")
    net.add_nodes(alice, bob, carol)

    eve = EquivocatingNode("Eve")
    net.add_byzantine(eve)

    # ── Round 1: token creation ───────────────────────────────────────────────
    section("Round 1 — Token creation")

    for node, amt in [(alice, 200), (bob, 100), (carol, 50)]:
        blk = node.create_block({"type": "create", "account": node.name.lower(), "amount": amt})
        net.broadcast_add(blk, node.name)
        print(f"  {node.name} creates {amt} tokens — block {blk.identity.short()!r}"
              f"  known_byz={len(blk.known_byz)}")

    net.deliver_pending()
    net.print_status("After Round 1")

    # ── Round 2: sequential transfer ──────────────────────────────────────────
    section("Round 2 — Sequential transfer Alice -> Bob (50 tokens)")

    blk = alice.create_block({"type": "transfer", "from": "alice", "to": "bob", "amount": 50})
    net.broadcast_add(blk, "Alice")
    print(f"  Alice transfers 50 -> Bob — block {blk.identity.short()!r}")

    net.deliver_pending()

    blk = bob.create_block({"type": "ack", "from": "alice", "to": "bob"})
    net.broadcast_add(blk, "Bob")
    print(f"  Bob acks receipt      — block {blk.identity.short()!r}")

    net.deliver_pending()
    net.print_status("After Round 2")

    # ── Round 3: Eve equivocates ──────────────────────────────────────────────
    section("Round 3 — Eve equivocates (double-spend)")
    print("""
  Eve creates blockA and blockB with same parents, nonce N and N+1.
  Both stored in Eve's B[bz].
  Both sent to ALL Nodes \\ {Eve}  (TLA+: ByzantineEquivocate).
    """)

    frontier = frozenset(eve.blocklace.maximal_blocks())
    fork_a, fork_b = eve.equivocate(
        {"type": "transfer", "from": "eve", "to": "alice", "amount": 80},
        {"type": "transfer", "from": "eve", "to": "bob",   "amount": 80},
        predecessors=frozenset(),
    )
    print(f"  fork_A: {fork_a.identity.short()!r}  known_byz={list(fork_a.known_byz)}")
    print(f"  fork_B: {fork_b.identity.short()!r}  known_byz={list(fork_b.known_byz)}")

    # TLA+: ByzantineEquivocate sends both to ALL Nodes \ {bz}
    net.broadcast_equivocate(fork_a, fork_b, "Eve")
    net.deliver_pending()
    net.print_status("After Eve equivocates")

    # ── Detection check ───────────────────────────────────────────────────────
    section("Byzantine detection (TLA+: EquivocationDetection)")
    for node in [alice, bob, carol]:
        eve_detected = eve.node_id in node.byz
        print(f"  {node.name}: byz={len(node.byz)}  Eve detected={eve_detected}")

    # ── Round 4: post-detection honest blocks ─────────────────────────────────
    section("Round 4 — Post-detection blocks (known_byz = {Eve})")
    print("  Honest nodes create blocks with known_byz = byz[n] = {Eve}")

    for node in [alice, bob, carol]:
        blk = node.create_block({"type": "create", "account": node.name.lower(), "amount": 1})
        net.broadcast_add(blk, node.name)
        print(f"  {node.name}: block {blk.identity.short()!r}  known_byz={len(blk.known_byz)} nodes")

    net.deliver_pending()
    net.print_status("After Round 4")

    # ── FiniteHarm: Eve's new block rejected ──────────────────────────────────
    section("FiniteHarm — Eve's new block after detection")

    evil_new = eve.add_honest(
        {"type": "transfer", "from": "eve", "to": "carol", "amount": 9999},
        predecessors=frozenset({fork_a.identity}),
    )
    net.broadcast_byz_honest(evil_new, "Eve")
    net.deliver_pending()

    accepted_by = [
        n.name for n in [alice, bob, carol]
        if evil_new.identity in n.blocklace
    ]
    print(f"  Eve's new block {evil_new.identity.short()!r}")
    print(f"  Accepted by: {accepted_by or 'NONE'}  (expected: NONE)")

    # ── GOC-Ledger from POLog ─────────────────────────────────────────────────
    section("GOC-Ledger reconstructed from each node's POLog")
    for node in [alice, bob, carol]:
        print()
        print_ledger(build_ledger_from_polog(node), f"{node.name}'s ledger")

    # ── TLA+ invariants ───────────────────────────────────────────────────────
    section("TLA+ invariants")

    print(f"  {'Node':<10} {'ClosureAxiom':>14} {'VirtualChain':>14} "
          f"{'EqvcDetect':>12} {'ByzRepelling':>14} {'FiniteHarm':>12}")
    for node in [alice, bob, carol]:
        # ClosureAxiom: ∀ b ∈ B[n]: b.parents ⊆ SHSet(B[n])
        closure_ok = all(
            p in node.blocklace
            for b in node.blocklace.blocks.values()
            for p in b.predecessors
        )
        # VirtualChainAxiom: any two honest blocks from same author are comparable
        vchain_ok = True
        bids = list(node.blocklace.blocks.keys())
        for i in range(len(bids)):
            for j in range(i + 1, len(bids)):
                a, b = bids[i], bids[j]
                if a.node_id == b.node_id and a.node_id == node.node_id:
                    if not node.blocklace.precedes(a, b) and not node.blocklace.precedes(b, a):
                        vchain_ok = False
        # EquivocationDetection: Eqvc(B[n]) ⊆ byz[n]
        eqvc_ok = node.blocklace.equivocators() <= node.byz
        # ByzRepelling: no equivocating blocks in POLog
        polog = node.blocklace.polog()
        eqv_blocks = node.blocklace.equivocators()
        byz_rep_ok = all(
            bid.node_id not in eqv_blocks or
            not any(
                node.blocklace.concurrent(bid, bid2)
                for bid2 in node.blocklace.blocks
                if bid2 != bid and bid2.node_id == bid.node_id
            )
            for bid in polog
        )
        # FiniteHarm: Eve's post-detection block not in blocklace
        finite_harm_ok = evil_new.identity not in node.blocklace

        print(f"  {node.name:<10} {str(closure_ok):>14} {str(vchain_ok):>14} "
              f"{str(eqvc_ok):>12} {str(byz_rep_ok):>14} {str(finite_harm_ok):>12}")

    # ── Convergence ───────────────────────────────────────────────────────────
    section("Convergence (TLA+: POLogConvergence)")
    conv = net.convergence_check()
    pologs = {n.name: len(n.polog_payloads()) for n in [alice, bob, carol]}
    print(f"  POLog sizes: {pologs}")
    print(f"  Convergence: {'YES' if conv else 'NO'}")
    print()


if __name__ == "__main__":
    main()
