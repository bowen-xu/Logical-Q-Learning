from .nal import DesireV, TruthV, Truth_induction


class Desire:
    def __init__(self, desirev: DesireV = None):
        self.desirev: DesireV = desirev or DesireV(0.5, 0.0)
        self.best_solution: Belief = None

    def choose(self, other_desirev: DesireV):
        """Choose between current desirev and another one based on sharpness."""
        self.desirev.choose(other_desirev)

    def overwrite(self, other_desirev: DesireV):
        """Overwrite current desirev with another one."""
        self.desirev = DesireV(other_desirev.f, other_desirev.c)

    def revise(self, other_desirev: DesireV, c_max=0.99):
        """Revise current desirev with another one."""
        self.desirev.revise(other_desirev.f, other_desirev.c, c_max=c_max)

class Belief:
    def __init__(self, truthv: TruthV = None):
        self.truthv: TruthV = truthv or TruthV(0.5, 0.0)
        self.eternal = False
        self.t_occur = -1

    def induction(self, truthv1: TruthV, truthv2: TruthV):
        truthv = Truth_induction(truthv1, truthv2)
        self.truthv.revise(truthv.f, truthv.c)