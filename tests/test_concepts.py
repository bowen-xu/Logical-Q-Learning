import unittest
import sys

sys.path.insert(0, "/Users/bowenxu/Codes/Logical-Q-Learning")

from lql.concepts import Concept, Sequence, Schema, PredictiveImplication
from lql.elements import Desire, Belief
from lql.nal import TruthV, DesireV


class TestConcept(unittest.TestCase):
    def test_create_concept(self):
        concept = Concept("apple")
        assert concept.value == "apple"
        assert isinstance(concept.desire, Desire)
        assert isinstance(concept.belief, Belief)

    def test_concept_has_upper_sequences(self):
        concept = Concept("test")
        assert hasattr(concept, "upper_sequences")
        assert len(concept.upper_sequences) == 0

    def test_concept_has_antecedents_consequents(self):
        concept = Concept("test")
        # Current implementation tracks schemas via in_schemas/out_schemas sets.
        assert hasattr(concept, "in_schemas")
        assert hasattr(concept, "out_schemas")
        assert len(concept.in_schemas) == 0
        assert len(concept.out_schemas) == 0

    def test_concept_hash(self):
        c1 = Concept("test")
        c2 = Concept("test")
        c3 = Concept("other")
        assert hash(c1) == hash(c2)
        assert hash(c1) != hash(c3)

    def test_concept_equality(self):
        c1 = Concept("test")
        c2 = Concept("test")
        c3 = Concept("other")
        assert c1 == c2
        assert c1 != c3
        assert c1 != "test"

    def test_concept_term_str(self):
        concept = Concept("hello")
        assert concept.term_str() == "hello"

    def test_concept_repr(self):
        concept = Concept("x")
        assert repr(concept) == "x"


class TestSequence(unittest.TestCase):
    def test_create_sequence(self):
        c1 = Concept("a")
        c2 = Concept("b")
        seq = Sequence(c1, c2)
        assert seq.components == (c1, c2)

    def test_sequence_adds_to_upper_sequences(self):
        c1 = Concept("a")
        c2 = Concept("b")
        seq = Sequence(c1, c2)
        assert seq in c1.upper_sequences

    def test_sequence_multiple_upper_sequences(self):
        c1 = Concept("a")
        c2 = Concept("b")
        c3 = Concept("c")
        seq1 = Sequence(c1, c2)
        seq2 = Sequence(c1, c3)
        seq3 = Sequence(c1, c2, c3)
        assert seq1 in c1.upper_sequences
        assert seq2 in c1.upper_sequences
        assert seq3 in c1.upper_sequences

    def test_sequence_len(self):
        c1 = Concept("a")
        c2 = Concept("b")
        c3 = Concept("c")
        seq = Sequence(c1, c2, c3)
        assert len(seq) == 3

    def test_sequence_hash(self):
        c1 = Concept("a")
        c2 = Concept("b")
        seq1 = Sequence(c1, c2)
        seq2 = Sequence(c1, c2)
        assert hash(seq1) == hash(seq2)

    def test_sequence_equality(self):
        c1 = Concept("a")
        c2 = Concept("b")
        seq1 = Sequence(c1, c2)
        seq2 = Sequence(c1, c2)
        seq3 = Sequence(c2, c1)
        assert seq1 == seq2
        assert seq1 != seq3

    def test_sequence_term_str(self):
        c1 = Concept("a")
        c2 = Concept("b")
        seq = Sequence(c1, c2)
        assert seq.term_str() == "(&/, a, b)"


class TestSchema(unittest.TestCase):
    def test_create_schema(self):
        c1 = Concept("state")
        c2 = Concept("next_state")
        schema = Schema(c1, c2)
        assert schema.antecedent == c1
        assert schema.consequent == c2
        # Schema belief starts as None unless an explicit truth is provided.
        assert schema.belief is None

    def test_schema_creates_relations(self):
        c1 = Concept("s")
        c2 = Concept("s'")
        schema = Schema(c1, c2)
        assert schema in c1.out_schemas
        assert schema in c2.in_schemas

    def test_schema_hash(self):
        c1 = Concept("a")
        c2 = Concept("b")
        s1 = Schema(c1, c2)
        s2 = Schema(c1, c2)
        assert hash(s1) == hash(s2)

    def test_schema_equality(self):
        c1 = Concept("a")
        c2 = Concept("b")
        c3 = Concept("c")
        s1 = Schema(c1, c2)
        s2 = Schema(c1, c2)
        s3 = Schema(c1, c3)
        assert s1 == s2
        assert s1 != s3


class TestTruthV(unittest.TestCase):
    def test_truthv_creation(self):
        tv = TruthV(0.8, 0.9)
        assert tv.f == 0.8
        assert tv.c == 0.9

    def test_truthv_e_property(self):
        tv = TruthV(1.0, 1.0)
        assert tv.e == 1.0
        tv2 = TruthV(0.0, 1.0)
        assert tv2.e == 0.0
        tv3 = TruthV(0.5, 1.0)
        assert tv3.e == 0.5

    def test_truthv_revision(self):
        tv = TruthV(0.5, 0.5)
        tv.revise(0.9, 0.8)
        assert tv.f > 0.5
        assert tv.c > 0.5

    def test_truthv_sharpness(self):
        tv1 = TruthV(1.0, 1.0)
        tv2 = TruthV(0.6, 1.0)
        assert TruthV.sharpness(tv1) == 1.0
        assert 0 < TruthV.sharpness(tv2) < 1.0

    def test_truthv_repr(self):
        tv = TruthV(0.75, 0.5)
        r = repr(tv)
        assert "%0.75%" in r or "%0.75," in r
        assert "%0.50%" in r or "0.50%" in r


class TestDesire(unittest.TestCase):
    def test_desire_creation(self):
        desire = Desire()
        assert isinstance(desire.desirev, DesireV)
        assert desire.best_solution is None

    def test_desire_with_value(self):
        dv = DesireV(0.8, 0.9)
        desire = Desire(dv)
        assert desire.desirev.f == 0.8
        assert desire.desirev.c == 0.9


class TestBelief(unittest.TestCase):
    def test_belief_creation(self):
        belief = Belief()
        assert isinstance(belief.truthv, TruthV)
        assert belief.eternal is False
        assert belief.t_occur == -1

    def test_belief_with_truthv(self):
        tv = TruthV(0.7, 0.85)
        belief = Belief(tv)
        assert belief.truthv.f == 0.7
        assert belief.truthv.c == 0.85
