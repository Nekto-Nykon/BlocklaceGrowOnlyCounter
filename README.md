# Blocklace + GOC-Ledger

A Python implementation of a **Byzantine-repelling Blocklace CRDT** and a **Grow-Only-Counter (GOC) Ledger** built on top of it. The project shows how a replicated ledger can stay safe in a peer-to-peer network without consensus, even when some nodes cheat (double-spend, equivocate, send malformed blocks or collude).

The implementation follows the Blocklace and GOC-Ledger papers and mirrors a TLA+ specification (`Newtry`) of the protocol: functions and comments in the code reference the corresponding TLA+ actions (`Add`, `Receive`, `CanAccept`, `POLog`, ...).

## Key ideas

- **Blocklace** – a DAG of signed blocks. Each block holds a payload, pointers to its predecessors and the set of Byzantine nodes known to its creator (`known_byz`). Block identity is the SHA-256 hash of the content, signed with the creator's ECDSA (P-256) key.
- **Equivocation detection** – a node that creates two concurrent (incomparable) blocks is detected as Byzantine. Both blocks are kept as evidence but excluded from the *PO-Log* (the partially ordered log of valid blocks), and later blocks from that node are rejected (*finite harm*).
- **GOC-Ledger** – every account is a set of grow-only counters (`created`, `burned`, `given`, `acked`). Merge is an element-wise `max`, so replicas converge automatically (strong eventual consistency). The safety invariant (total token conservation) holds even when a double-spend happens concurrently.

## Project structure

```
src/
  crypto.py       ECDSA keys, signing, SHA-256, canonical JSON
  block.py        BlockId, BlockContent, Block (signed, with known_byz)
  blocklace.py    Blocklace CRDT: causal order, equivocation, PO-Log, byzantine-repelling check
  account.py      Account CRDT (grow-only counters, merge, partial order)
  ledger.py       Ledger CRDT (grow-only dictionary of accounts, safety invariant)
  node.py         Honest P2P node (Add / Receive / CanAccept)
  byzantine.py    Faulty nodes: equivocating, malformed-block and colluding nodes
  simulation.py   Simulated P2P network with persistent messages
examples/         Three runnable demos
tests/            81 tests (unit + property-based)
```

## Getting started

Requires Python 3.9+.

```bash
git clone https://github.com/Nekto-Nykon/BlocklaceGrowOnlyCounter.git
cd BlocklaceGrowOnlyCounter
pip install cryptography pytest hypothesis
```

### Run the demos

```bash
python examples/demo_ledger.py            # GOC-Ledger alone: sequential vs. concurrent (double-spend) transfers
python examples/demo_blocklace.py         # 3 honest nodes + 1 equivocating node
python examples/demo_blocklace_ledger.py  # full scenario: ledger rebuilt from each node's PO-Log
```

In the full scenario, Alice, Bob and Carol are honest and Eve equivocates. All honest nodes detect Eve, exclude her forks from the PO-Log, and end up with identical ledgers and converged PO-Logs.

### Run the tests

```bash
python -m pytest
```

| File | What it checks |
|---|---|
| `test_blocklace.py` | block structure, causal order, well-formedness, equivocation detection, PO-Log |
| `test_byzantine.py` | equivocating, malformed and colluding nodes |
| `test_ledger.py` | account/ledger operations, merge, safety invariant |
| `test_probabilistic.py` | property-based tests (Hypothesis): CRDT laws (commutativity, associativity, idempotency), safety invariant, bounded double-spend exposure, honest-network convergence |

## Tech stack

Python, `cryptography` (ECDSA / SHA-256), pytest, Hypothesis.

## References

- <https://arxiv.org/abs/2402.08068>The Blocklace: A Byzantine-repelling and Universal Conflict-free Replicated Data Type (Paulo Sérgio Almeida, Ehud Shapiro)
- <https://arxiv.org/abs/2305.16976>GOC-Ledger: State-based Conflict-Free Replicated Ledger from Grow-Only Counters (Erick Lavoie)
