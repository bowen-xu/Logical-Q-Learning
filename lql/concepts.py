from .nal import DesireV, TruthV
from .elements import Desire, Belief
from weakref import WeakSet


class Concept:
    def __init__(self, value):
        self.value = value
        self.desire = Desire()
        self.belief = Belief()
        self.upper_sequences = WeakSet[Sequence]()
        self.in_schemas = WeakSet[Schema]()
        self.out_schemas = WeakSet[Schema]()

    def __eq__(self, other):
        if not isinstance(other, Concept):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    @staticmethod
    def compute_hash(value):
        return hash(value)

    def __repr__(self):
        return f"{self.value}"

    def term_str(self):
        return f"{self.value}"


class Sequence(Concept):
    def __init__(self, *args: Concept):
        Concept.__init__(self, tuple(args))
        c0 = args[0]
        c0.upper_sequences.add(self)

    @property
    def components(self) -> tuple[Concept]:
        return self.value

    def __len__(self):
        return len(self.components)

    def __hash__(self):
        return hash(self.components)

    @staticmethod
    def compute_hash(*components):
        return hash(tuple(components))

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return False
        if hash(self) != hash(other):
            return False
        return self.components == other.components

    def __repr__(self):
        return f"{self.term_str()} {self.desire.desirev}!"

    def term_str(self):
        return f"(&/, {', '.join(str(c.value) for c in self.components)})"

    def evaluate_desire(self) -> float:
        return self.evaluate_desire_by_e()

    def evaluate_desire_by_e(self):
        return self.desire.desirev.e


class PredictiveImplication:
    def __init__(
        self, antecedent: Sequence | Concept, consequent: Concept, truth: TruthV = None
    ):
        self.antecedent = antecedent
        self.consequent = consequent
        antecedent.out_schemas.add(self)
        consequent.in_schemas.add(self)

        self.belief = Belief(truth) if truth is not None else None

    def __hash__(self):
        return hash((self.antecedent, self.consequent))

    @staticmethod
    def compute_hash(antecedent, consequent):
        return hash((antecedent, consequent))

    def __eq__(self, other):
        if not isinstance(other, PredictiveImplication):
            return False
        if hash(self) != hash(other):
            return False
        return (self.antecedent, self.consequent) == (
            other.antecedent,
            other.consequent,
        )

    def __repr__(self):
        return f"({self.antecedent.term_str()} =/> {self.consequent.term_str()}) {self.belief.truthv}"


Schema = PredictiveImplication
