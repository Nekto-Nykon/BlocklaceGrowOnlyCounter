"""
Demo: Byzantine-Repelling Blocklace (aligned with TLA+ Newtry module)

Scenario
--------
Three honest nodes (Alice, Bob, Carol).
One Byzantine equivocating node (Eve) sends BOTH conflicting blocks to ALL nodes
(TLA+: ByzantineEquivocate sends to Nodes \\ {bz}).

Expected outcome:
  - All honest nodes detect Eve via Eqvc check in Receive.
  - Both forks accepted as evidence, both excluded from POLog.
  - Eve post-detection blocks rejected (FiniteHarm).
  - POLogs converge across all honest nodes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.byzantine import EquivocatingNode, MalformedBlockNode
from src.node import HonestNode
from src.simulation import Network


def section(title):
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def main():
    print("=" * 60)
    print("  Byzantine-Repelling Blocklace Demo (Newtry TLA+ model)")
    print("=" * 60)

    net = Network()
    alice = HonestNode("Alice")
    bob   = HonestNode("Bob")
    carol = HonestNode("Carol")
    net.add_nodes(alice, bob, carol)

    eve = EquivocatingNode("Eve")
    net.add_byzantine(eve)

    # ── Round 1: honest genesis blocks ───────────────────────────────────────
    section("Round 1 - Honest genesis blocks")
    for node, msg in [(alice, "alice_genesis"), (bob, "bob_genesis"), (carol, "carol_genesis")]:
        blk = node.create_block(msg)
        net.broadcast_add(blk, node.name)
        print(f"  {node.name} created block: {blk.identity.short()!r}  known_byz={len(blk.known_byz)}")

    net.deliver_pending()
    net.print_status("After Round 1")

    # ── Round 2: Eve equivocates ──────────────────────────────────────────────
    section("Round 2 - Eve equivocates (both forks to all nodes)")
    fork_a, fork_b = eve.equivocate("evil_A", "evil_B", frozenset())
    print(f"  Eve fork A: {fork_a.identity.short()!r}")
    print(f"  Eve fork B: {fork_b.identity.short()!r}")
    net.broadcast_equivocate(fork_a, fork_b, "Eve")

    net.deliver_pending()
    net.print_status("After Eve equivocation")

    for node in [alice, bob, carol]:
        print(f"  {node.name}: eve_detected={eve.node_id in node.byz}  "
              f"fork_a_in_B={fork_a.identity in node.blocklace}  "
              f"fork_b_in_B={fork_b.identity in node.blocklace}  "
              f"fork_a_in_polog={fork_a.identity in node.blocklace.polog()}  "
              f"fork_b_in_polog={fork_b.identity in node.blocklace.polog()}")

    # ── Round 3: honest post-detection blocks ─────────────────────────────────
    section("Round 3 - Honest blocks after detection (known_byz = {Eve})")
    for node, msg in [
        (alice, "alice_post"),
        (bob,   "bob_post"),
        (carol, "carol_post"),
    ]:
        blk = node.create_block(msg)
        net.broadcast_add(blk, node.name)
        print(f"  {node.name}: block {blk.identity.short()!r}  known_byz={len(blk.known_byz)}")

    net.deliver_pending()
    net.print_status("After Round 3")

    # ── Round 4: FiniteHarm ───────────────────────────────────────────────────
    section("Round 4 - FiniteHarm: post-detection Eve block")
    eve_late = eve.add_honest("evil_late", frozenset({fork_a.identity}))
    net.broadcast_byz_honest(eve_late, "Eve")
    net.deliver_pending()

    accepted = [
        name for name, node in net.honest_nodes.items()
        if eve_late.identity in node.blocklace
    ]
    print(f"  Eve late block accepted by: {accepted or 'NONE'} (expected: NONE)")

    # ── Analysis ──────────────────────────────────────────────────────────────
    section("Analysis — TLA+ invariants")
    for node in [alice, bob, carol]:
        polog = node.blocklace.polog()
        payloads = set(node.polog_payloads().values())
        closure_ok = all(
            p in node.blocklace
            for b in node.blocklace.blocks.values()
            for p in b.predecessors
        )
        eqvc_ok = node.blocklace.equivocators() <= node.byz
        print(f"\n  {node.name}:")
        print(f"    ClosureAxiom      : {closure_ok}")
        print(f"    EqvcDetection     : {eqvc_ok}")
        print(f"    Eve detected      : {eve.node_id in node.byz}")
        print(f"    POLog size        : {len(polog)}")
        print(f"    evil_late in polog: {'evil_late' in payloads}")
        print(f"    FiniteHarm        : {eve_late.identity not in node.blocklace}")

    print("\n  POLog convergence:", end=" ")
    pologs = [frozenset(n.polog_payloads().keys()) for n in [alice, bob, carol]]
    print("YES" if len(set(pologs)) == 1 else "NO")
    print()


if __name__ == "__main__":
    main()