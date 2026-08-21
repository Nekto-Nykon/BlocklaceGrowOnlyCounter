"""
GOC Ledger CRDT — §2.2 of the GOC-Ledger paper.

A ledger is a grow-only dictionary of Account CRDTs.  Adding an account
either inserts it (if new) or merges it with the stored replica.

Safety invariant (§2.4):
  Σ balance(A≥0)  ≤  Σ created(C) − Σ burned(A) + Σ |balance(A<0)|

This always holds, even in the presence of concurrent double-spending, as
long as the ledger is a transitive closure over acknowledged senders.
"""
from __future__ import annotations

from typing import Dict, Optional, Set

from .account import Account


class Ledger:
    """Replicated ledger — a grow-only dictionary of Account CRDTs."""

    def __init__(self) -> None:
        self._accounts: Dict[str, Account] = {}

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, account: Account) -> None:
        """
        add(L, A) — Alg. 5.
        Insert *account* or merge with existing replica for the same id.
        """
        aid = account.aid
        if aid not in self._accounts:
            self._accounts[aid] = account._copy()
        else:
            self._accounts[aid] = self._accounts[aid].merge(account)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get(self, aid: str) -> Optional[Account]:
        return self._accounts.get(aid)

    def balances(self) -> Dict[str, float]:
        """balances(L) — Alg. 5."""
        return {aid: acc.balance() for aid, acc in self._accounts.items()}

    def account_ids(self) -> Set[str]:
        return set(self._accounts.keys())

    # ── CRDT join (⊔L) ────────────────────────────────────────────────────────

    def merge(self, other: "Ledger") -> "Ledger":
        """⊔L: merge two ledger replicas."""
        result = Ledger()
        for aid in set(self._accounts) | set(other._accounts):
            if aid in self._accounts and aid in other._accounts:
                result._accounts[aid] = self._accounts[aid].merge(other._accounts[aid])
            elif aid in self._accounts:
                result._accounts[aid] = self._accounts[aid]._copy()
            else:
                result._accounts[aid] = other._accounts[aid]._copy()
        return result

    # ── Partial order (≤L) ────────────────────────────────────────────────────

    def leq(self, other: "Ledger") -> bool:
        for aid, acc in self._accounts.items():
            if aid not in other._accounts:
                return False
            if not acc.leq(other._accounts[aid]):
                return False
        return True

    # ── Safety property §2.4 ─────────────────────────────────────────────────

    def safety_holds(self) -> bool:
        """
        Check the token-conservation safety invariant:
          Σ balance(A≥0) ≤ Σ created − Σ burned + Σ |balance(A<0)|

        Overspending (negative balances) can only arise from concurrent
        updates to the same account; correct sequential updates always
        maintain non-negative balances (proven by induction in §3.5.1).
        """
        pos = sum(b for b in self.balances().values() if b >= 0)
        neg = sum(-b for b in self.balances().values() if b < 0)
        total_created = sum(a.created for a in self._accounts.values())
        total_burned = sum(a.burned for a in self._accounts.values())
        # 1e-9 tolerance absorbs floating-point rounding that accumulates
        # across multiple give/ack/merge operations on float64 counters.
        return pos <= total_created - total_burned + neg + 1e-9

    def double_spend_exposure(self) -> float:
        """
        The total negative balance across all accounts — this is the maximum
        possible impact of double-spending on correct nodes.  The GOC-Ledger
        bounds this to concurrent operations only; the Blocklace integration
        (via Byzantine-repelling dissemination) further limits which concurrent
        operations can reach correct nodes.
        """
        return sum(-b for b in self.balances().values() if b < 0)

    def total_created(self) -> float:
        return sum(a.created for a in self._accounts.values())

    def total_burned(self) -> float:
        return sum(a.burned for a in self._accounts.values())

    def __repr__(self) -> str:
        bals = ", ".join(f"{aid}={b:.2f}" for aid, b in self.balances().items())
        return f"Ledger({bals})"
