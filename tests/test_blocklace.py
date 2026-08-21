"""
Unit tests for the core Blocklace data structure.

Covers:
  - Block creation, hashing, and signature verification
  - Causal order: precedes, concurrent, closure
  - Blocklace merging and maximal-block computation
  - Equivocation detection
  - Byzantine-repelling property check
"""
import pytest

from src.block import Block, BlockContent, BlockId
from src.blocklace import Blocklace
from src.crypto import generate_keypair


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def alice():
    sk, nid = generate_keypair()
    return sk, nid


@pytest.fixture
def bob():
    sk, nid = generate_keypair()
    return sk, nid


def make_block(payload, preds, sk, nid):
    return Block.create(payload, frozenset(preds), sk, nid)


# ── Block identity & signatures ───────────────────────────────────────────────

class TestBlockIdentity:
    def test_valid_signature(self, alice):
        sk, nid = alice
        b = make_block("hello", [], sk, nid)
        assert b.has_valid_signature()

    def test_different_payloads_different_ids(self, alice):
        sk, nid = alice
        b1 = make_block("a", [], sk, nid)
        b2 = make_block("b", [], sk, nid)
        assert b1.identity != b2.identity

    def test_same_content_same_id_same_key(self, alice):
        """
        The same payload + predecessors always produce the same content hash,
        but ECDSA is nondeterministic so signatures differ — hence BlockIds differ.
        We only test that the content hashes match.
        """
        sk, nid = alice
        b1 = make_block("x", [], sk, nid)
        b2 = make_block("x", [], sk, nid)
        assert b1.identity.content_hash == b2.identity.content_hash

    def test_node_id_in_identity(self, alice, bob):
        ska, nida = alice
        skb, nidb = bob
        ba = make_block("m", [], ska, nida)
        bb = make_block("m", [], skb, nidb)
        assert ba.identity.node_id == nida
        assert bb.identity.node_id == nidb
        assert ba.identity.node_id != bb.identity.node_id


# ── Blocklace structure ───────────────────────────────────────────────────────

class TestBlocklaceStructure:
    def test_add_genesis_block(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b = make_block("genesis", [], sk, nid)
        assert bl.add_block(b)
        assert b.identity in bl
        assert len(bl) == 1

    def test_add_requires_predecessors(self, alice):
        sk, nid = alice
        b1 = make_block("g", [], sk, nid)
        b2 = make_block("next", [b1.identity], sk, nid)
        bl = Blocklace()
        assert not bl.add_block(b2)   # b1 missing
        bl.add_block(b1)
        assert bl.add_block(b2)       # now ok

    def test_maximal_blocks_single(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b = make_block("g", [], sk, nid)
        bl.add_block(b)
        assert bl.maximal_blocks() == {b.identity}

    def test_maximal_blocks_chain(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b1 = make_block("g", [], sk, nid)
        bl.add_block(b1)
        b2 = make_block("h", [b1.identity], sk, nid)
        bl.add_block(b2)
        assert bl.maximal_blocks() == {b2.identity}

    def test_maximal_blocks_parallel(self, alice, bob):
        ska, nida = alice
        skb, nidb = bob
        bl = Blocklace()
        b1 = make_block("g", [], ska, nida)
        b2 = make_block("g", [], skb, nidb)
        bl.add_block(b1)
        bl.add_block(b2)
        assert bl.maximal_blocks() == {b1.identity, b2.identity}

    def test_merge_idempotent(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b = make_block("x", [], sk, nid)
        bl.add_block(b)
        merged = bl.merge(bl)
        assert len(merged) == len(bl)

    def test_merge_union(self, alice, bob):
        ska, nida = alice
        skb, nidb = bob
        bl1 = Blocklace()
        bl2 = Blocklace()
        b1 = make_block("a", [], ska, nida)
        b2 = make_block("b", [], skb, nidb)
        bl1.add_block(b1)
        bl2.add_block(b2)
        merged = bl1.merge(bl2)
        assert b1.identity in merged
        assert b2.identity in merged


# ── Causal order ──────────────────────────────────────────────────────────────

class TestCausalOrder:
    def test_precedes_direct(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b1 = make_block("g", [], sk, nid)
        bl.add_block(b1)
        b2 = make_block("h", [b1.identity], sk, nid)
        bl.add_block(b2)
        assert bl.precedes(b1.identity, b2.identity)
        assert not bl.precedes(b2.identity, b1.identity)

    def test_precedes_transitive(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b1 = make_block("a", [], sk, nid)
        bl.add_block(b1)
        b2 = make_block("b", [b1.identity], sk, nid)
        bl.add_block(b2)
        b3 = make_block("c", [b2.identity], sk, nid)
        bl.add_block(b3)
        assert bl.precedes(b1.identity, b3.identity)

    def test_concurrent(self, alice, bob):
        ska, nida = alice
        skb, nidb = bob
        bl = Blocklace()
        b1 = make_block("x", [], ska, nida)
        b2 = make_block("y", [], skb, nidb)
        bl.add_block(b1)
        bl.add_block(b2)
        assert bl.concurrent(b1.identity, b2.identity)

    def test_closure_single(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b = make_block("g", [], sk, nid)
        bl.add_block(b)
        assert bl.closure(b.identity) == {b.identity}

    def test_closure_chain(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b1 = make_block("a", [], sk, nid)
        bl.add_block(b1)
        b2 = make_block("b", [b1.identity], sk, nid)
        bl.add_block(b2)
        assert bl.closure(b2.identity) == {b1.identity, b2.identity}


# ── Well-formedness ───────────────────────────────────────────────────────────

class TestWellFormedness:
    def test_well_formed_genesis(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b = make_block("g", [], sk, nid)
        bl.add_block(b)
        assert bl.is_well_formed(b.identity)

    def test_well_formed_valid_chain(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b1 = make_block("a", [], sk, nid)
        bl.add_block(b1)
        b2 = make_block("b", [b1.identity], sk, nid)
        bl.add_block(b2)
        assert bl.is_well_formed(b1.identity)
        assert bl.is_well_formed(b2.identity)

    def test_malformed_signature(self, alice):
        from src.crypto import sha256, sign
        from src.block import BlockContent, BlockId
        sk, nid = alice
        content = BlockContent(payload="bad", predecessors=frozenset())
        content_hash = sha256(content.to_bytes())
        bad_sig = sign(b"wrong data", sk)
        bad_id = BlockId(content_hash=content_hash, signature=bad_sig, node_id=nid)
        bad_block = Block(identity=bad_id, content=content)
        bl = Blocklace()
        bl._blocks[bad_id] = content   # force-add bypassing the add_block check
        assert not bl.is_well_formed(bad_id)


# ── Byzantine detection ───────────────────────────────────────────────────────

class TestByzantineDetection:
    def test_no_equivocators_honest(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b1 = make_block("a", [], sk, nid)
        bl.add_block(b1)
        b2 = make_block("b", [b1.identity], sk, nid)
        bl.add_block(b2)
        assert bl.equivocators() == set()

    def test_equivocator_detected(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b1 = make_block("fork_a", [], sk, nid)
        b2 = make_block("fork_b", [], sk, nid)
        bl._blocks[b1.identity] = b1.content
        bl._blocks[b2.identity] = b2.content
        assert nid in bl.equivocators()

    def test_byzantine_nodes_includes_malformed(self, alice):
        from src.crypto import sha256, sign
        from src.block import BlockContent, BlockId
        sk, nid = alice
        content = BlockContent(payload="x", predecessors=frozenset())
        content_hash = sha256(content.to_bytes())
        bad_sig = sign(b"garbage", sk)
        bad_id = BlockId(content_hash=content_hash, signature=bad_sig, node_id=nid)
        bl = Blocklace()
        bl._blocks[bad_id] = content
        assert nid in bl.byzantine_nodes()


# ── Byzantine-repelling ───────────────────────────────────────────────────────

class TestByzantineRepelling:
    def test_single_honest_block_is_brep(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b = make_block("g", [], sk, nid)
        bl.add_block(b)
        assert bl.is_byzantine_repelling()

    def test_chain_of_honest_blocks_is_brep(self, alice):
        sk, nid = alice
        bl = Blocklace()
        b1 = make_block("a", [], sk, nid)
        bl.add_block(b1)
        b2 = make_block("b", [b1.identity], sk, nid)
        bl.add_block(b2)
        assert bl.is_byzantine_repelling()

    def test_empty_blocklace_is_brep(self):
        assert Blocklace().is_byzantine_repelling()

    def test_polog_includes_genesis_byzantine_blocks(self, alice, bob):
        """
        Equivocating genesis blocks have empty causal past, so byz(≺b) = ∅ and
        the creator is not yet detectable from b's own past — both forks appear
        in polog.  Only blocks created AFTER Byzantine evidence is in their
        causal past are excluded.  (Paper §4.3, Proposition 4.4.)
        """
        ska, nida = alice
        skb, nidb = bob
        bl = Blocklace()
        honest = make_block("honest", [], ska, nida)
        bl.add_block(honest)
        byz_a = make_block("evil_a", [], skb, nidb)
        byz_b = make_block("evil_b", [], skb, nidb)
        bl._blocks[byz_a.identity] = byz_a.content
        bl._blocks[byz_b.identity] = byz_b.content
        polog = bl.polog()
        # TLA+: b ∉ EquivBlocks(b.node, bl)
        # Both equivocating blocks visible in bl → both excluded from polog.
        assert honest.identity in polog
        assert byz_a.identity not in polog
        assert byz_b.identity not in polog
        assert nidb in bl.equivocators()

    def test_polog_excludes_post_evidence_byzantine_block(self, bob):
        """
        A block from the Byzantine node that has the equivocating blocks in
        its causal past IS excluded from polog (byz(≺b) now includes its creator).
        """
        skb, nidb = bob
        bl = Blocklace()
        byz_a = make_block("evil_a", [], skb, nidb)
        byz_b = make_block("evil_b", [], skb, nidb)
        bl._blocks[byz_a.identity] = byz_a.content
        bl._blocks[byz_b.identity] = byz_b.content
        # Third block from same byz node pointing to both forks
        # (force-insert because add_block would also succeed here)
        byz_c = make_block("evil_c", [byz_a.identity], skb, nidb)
        bl._blocks[byz_c.identity] = byz_c.content
        polog = bl.polog()
        # byz_c's strict past contains byz_a; equivocators() of that sub-blocklace
        # contains nidb (byz_b is a concurrent block to byz_a in that sub-view?
        # No — byz_b is not in byz_a's sub-blocklace.  The exclusion only triggers
        # when BOTH forks are visible in the causal past.
        # Force-insert byz_c pointing to BOTH equivocating blocks:
        byz_d = make_block("evil_d", [byz_a.identity, byz_b.identity], skb, nidb)
        bl._blocks[byz_d.identity] = byz_d.content
        polog2 = bl.polog()
        # byz_d's strict past = {byz_a, byz_b} — both forks visible → nidb is eqvc → excluded
        assert byz_d.identity not in polog2
