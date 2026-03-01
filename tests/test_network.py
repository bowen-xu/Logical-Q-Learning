import unittest
import sys

sys.path.insert(0, "/Users/bowenxu/Codes/Logical-Q-Learning")

from lql.network import ConceptNetwork
from lql.concepts import Concept, Sequence, Schema
from lql.nal import TruthV


class TestConceptNetwork(unittest.TestCase):
    def test_create_network(self):
        network = ConceptNetwork()
        assert network.concepts == {}
        assert network.schemas == {}
        assert network.sequences == {}

    def test_ensure_concept_new(self):
        network = ConceptNetwork()
        concept = network.ensure_concept("apple")
        assert concept.value == "apple"
        assert "apple" in [c.value for c in network.concepts.values()]

    def test_ensure_concept_existing(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("test")
        c2 = network.ensure_concept("test")
        assert c1 is c2

    def test_ensure_schema_new(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("s1")
        c2 = network.ensure_concept("s2")
        schema = network.ensure_schema(c1, c2)
        assert schema.antecedent == c1
        assert schema.consequent == c2

    def test_ensure_schema_existing(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("a")
        c2 = network.ensure_concept("b")
        s1 = network.ensure_schema(c1, c2)
        s2 = network.ensure_schema(c1, c2)
        assert s1 is s2

    def test_ensure_sequence_new(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("c1")
        c2 = network.ensure_concept("c2")
        seq = network.ensure_sequence(c1, c2)
        assert len(seq.components) == 2
        assert c1 in seq.components
        assert c2 in seq.components

    def test_ensure_sequence_existing(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("a")
        c2 = network.ensure_concept("b")
        s1 = network.ensure_sequence(c1, c2)
        s2 = network.ensure_sequence(c1, c2)
        assert s1 is s2

    def test_ensure_sequence_single_component(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("only")
        seq = network.ensure_sequence(c1)
        assert len(seq.components) == 1

    def test_ensure_sequence_multiple_components(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("a")
        c2 = network.ensure_concept("b")
        c3 = network.ensure_concept("c")
        seq = network.ensure_sequence(c1, c2, c3)
        assert len(seq.components) == 3

    def test_concepts_stored_by_hash(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("value1")
        c2 = network.ensure_concept("value1")
        assert len(network.concepts) == 1

    def test_schemas_stored_by_hash(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("s")
        c2 = network.ensure_concept("s'")
        network.ensure_schema(c1, c2)
        network.ensure_schema(c1, c2)
        assert len(network.schemas) == 1

    def test_sequences_stored_by_hash(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("a")
        c2 = network.ensure_concept("b")
        network.ensure_sequence(c1, c2)
        network.ensure_sequence(c1, c2)
        assert len(network.sequences) == 1

    def test_different_concepts_separate(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("apple")
        c2 = network.ensure_concept("banana")
        c3 = network.ensure_concept("apple")
        assert c1 is c3
        assert c1 is not c2
        assert len(network.concepts) == 2

    def test_schema_creates_bidirectional_relation(self):
        network = ConceptNetwork()
        c1 = network.ensure_concept("state")
        c2 = network.ensure_concept("next")
        schema = network.ensure_schema(c1, c2)
        assert schema in c1.out_schemas
        assert schema in c2.in_schemas
