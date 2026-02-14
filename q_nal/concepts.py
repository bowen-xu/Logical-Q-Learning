from .nal import DesireV, TruthV

from weakref import WeakSet


class Desire:
    def __init__(self, desirev: DesireV = None):
        self.desirev: DesireV = desirev or DesireV(0.5, 0.0)
        self.best_solution: Belief = None
        self.instant = True
        self.t_schedule = -1

class Belief:
    def __init__(self, truthv: TruthV = None):
        self.truthv: TruthV = truthv or TruthV(0.5, 0.0)
        self.eternal = False
        self.t_occur = -1

class Concept:
    def __init__(self, value):
        self.value = value
        self.desire = Desire()
        self.belief = Belief()

        self.antecedents: WeakSet[Schema] = WeakSet()
        self.consequents: WeakSet[Schema] = WeakSet()

    def __eq__(self, other):
        if not isinstance(other, Concept):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"{self.value}"
    
    def term_str(self):
        return f"{self.value}"


class Sequence(Concept):
    def __init__(self, *args: Concept):
        Concept.__init__(self, tuple(args))

    @property
    def components(self):
        return self.value

    def __len__(self):
        return len(self.components)

    def __hash__(self):
        return hash(self.components)

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return False
        if hash(self) != hash(other):
            return False
        return self.components == other.components

    def __repr__(self):
        return (
            f"{self.term_str()} {self.desire.desirev}!"
        )
    
    def term_str(self):
        return f"(&/, {', '.join(str(c.value) for c in self.components)})"



class PredictiveImplication:
    def __init__(
        self, antecedent: Sequence | Concept, consequent: Concept, truth: TruthV = None
    ):
        self.antecedent = antecedent
        self.consequent = consequent
        antecedent.consequents.add(self)
        consequent.antecedents.add(self)

        self.belief = Belief(truth)

    def __hash__(self):
        return hash((self.antecedent, self.consequent))

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
