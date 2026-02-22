from .nal import DesireV, TruthV
from .elements import Desire, Belief
from .concepts import Concept, Schema, Sequence


class ConceptNetwork:
    def __init__(self):
        self.concepts = dict[int, Concept]()  # atomic terms
        self.schemas = dict[int, Schema]()  # (S, A) =/> S'
        self.sequences = dict[int, Sequence]()  # (S, A)

    def ensure_concept(self, value) -> Concept:
        hashv = Concept.compute_hash(value)
        if hashv not in self.concepts:
            self.concepts[hashv] = Concept(value)
        return self.concepts[hashv]

    def ensure_schema(self, antecedent: Concept, consequent: Concept) -> Schema:
        hashv = Schema.compute_hash(antecedent, consequent)
        if hashv not in self.schemas:
            self.schemas[hashv] = Schema(antecedent, consequent)
        return self.schemas[hashv]

    def ensure_sequence(self, *components: Concept) -> Sequence:
        hashv = Sequence.compute_hash(components)
        if hashv not in self.sequences:
            self.sequences[hashv] = Sequence(*components)
        return self.sequences[hashv]
