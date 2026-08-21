"""
GOC Account CRDT — §2.1 of the GOC-Ledger paper.

Each account is composed of grow-only counters so that:
  - all operations are monotone (state only grows),
  - merge is element-wise max (commutative, associative, idempotent),
  - strong eventual consistency is guaranteed automatically.

Counter semantics (paper notation):
  created  A↑   total tokens ever created by this account
  burned   A↓   total tokens ever burned by this account
  given    A→   dict[id → amount]: total ever sent to each recipient
  acked    A←   dict[id → amount]: total ever acknowledged from each sender

balance = (created + Σ acked) − (burned + Σ given)
"""
from __future__ import annotations

from typing import Dict, Optional, Set


class Account:
    """Replicated account — a state-based CRDT over grow-only counters."""

    def __init__(self, aid: str, creator_ids: Optional[Set[str]] = None) -> None:
        self.aid = aid
        self._creator_ids: Set[str] = set(creator_ids) if creator_ids else set()
        self.created: float = 0.0
        self.burned: float = 0.0
        self.given: Dict[str, float] = {}   # A→
        self.acked: Dict[str, float] = {}   # A←

    # ── Query ─────────────────────────────────────────────────────────────────

    def balance(self) -> float:
        """balance(A) = (A↑ + Σ A←[id]) − (A↓ + Σ A→[id])   (Alg. 2)"""
        debits = self.created + sum(self.acked.values())
        credits = self.burned + sum(self.given.values())
        return debits - credits

    def unacked_from(self, sender: "Account") -> float:
        """
        unackedFrom(A, B): tokens sent by *sender* not yet acknowledged by self.
        B→[Aid] − A←[Bid]   (Alg. 2)
        """
        sent = sender.given.get(self.aid, 0.0)
        acked_so_far = self.acked.get(sender.aid, 0.0)
        return sent - acked_so_far

    # ── State-changing operations (each returns a new Account) ────────────────

    def create(self, amount: float) -> "Account":
        """
        create(A, amount): increase A↑ if this account is an authorised creator.
        """
        if self.aid not in self._creator_ids or amount <= 0:
            return self
        new = self._copy()
        new.created += amount
        return new

    def burn(self, amount: float) -> "Account":
        """burn(A, amount): increase A↓ if balance is sufficient."""
        if amount <= 0 or self.balance() < amount:
            return self
        new = self._copy()
        new.burned += amount
        return new

    def give_to(self, amount: float, recipient_id: str) -> "Account":
        """giveTo(A, amount, id): send *amount* tokens to *recipient_id*."""
        if amount <= 0 or self.balance() < amount:
            return self
        new = self._copy()
        new.given[recipient_id] = new.given.get(recipient_id, 0.0) + amount
        return new

    def ack_from(self, sender: "Account") -> "Account":
        """
        ackFrom(A, B): acknowledge all unacknowledged tokens from *sender*.
        Sets A←[Bid] = max(current, B→[Aid]).
        """
        if self.unacked_from(sender) <= 0:
            return self
        new = self._copy()
        new.acked[sender.aid] = max(
            new.acked.get(sender.aid, 0.0),
            sender.given.get(self.aid, 0.0),
        )
        return new

    # ── CRDT join (⊔A) ────────────────────────────────────────────────────────

    def merge(self, other: "Account") -> "Account":
        """
        ⊔A: least upper bound of two account states.
        Element-wise max of all grow-only counters.   (Alg. 3)
        """
        if self.aid != other.aid:
            raise ValueError(f"Cannot merge accounts with different ids: {self.aid!r} vs {other.aid!r}")
        result = Account(self.aid, self._creator_ids | other._creator_ids)
        result.created = max(self.created, other.created)
        result.burned = max(self.burned, other.burned)
        for rid in set(self.given) | set(other.given):
            result.given[rid] = max(self.given.get(rid, 0.0), other.given.get(rid, 0.0))
        for sid in set(self.acked) | set(other.acked):
            result.acked[sid] = max(self.acked.get(sid, 0.0), other.acked.get(sid, 0.0))
        return result

    # ── Partial order (≤A) ────────────────────────────────────────────────────

    def leq(self, other: "Account") -> bool:
        """≤A: self ≤ other iff every counter in self is ≤ the corresponding counter in other."""
        if self.aid != other.aid:
            return False
        if self.created > other.created or self.burned > other.burned:
            return False
        for rid, amt in self.given.items():
            if rid not in other.given or amt > other.given[rid]:
                return False
        for sid, amt in self.acked.items():
            if sid not in other.acked or amt > other.acked[sid]:
                return False
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _copy(self) -> "Account":
        new = Account(self.aid, self._creator_ids)
        new.created = self.created
        new.burned = self.burned
        new.given = dict(self.given)
        new.acked = dict(self.acked)
        return new

    def __repr__(self) -> str:
        return (
            f"Account({self.aid!r}, balance={self.balance():.2f}, "
            f"created={self.created}, burned={self.burned}, "
            f"given={self.given}, acked={self.acked})"
        )
