from __future__ import annotations

from functools import reduce
from statistics import mean
from operator import mul

fc_to_w_plus = lambda f, c, k: k * f * c / (1 - c)
c_to_w = lambda c, k: k * c / (1 - c)
fc_to_w_minus = lambda f, c, k: k * (1 - f) * c / (1 - c)

w_to_f = lambda w_plus, w: w_plus / max(w, 1e-5)
w_to_c = lambda w, k: w / (w + k)

Not = lambda x: (1 - x)

And = lambda *x: reduce(mul, x, 1)
Or = lambda *x: 1 - reduce(mul, (1 - xi for xi in x), 1)
Average = lambda *x: mean(x)


F_rev = F_revision = lambda w_p_1, w_p_2, w_m_1, w_m_2: (
    w_p_1 + w_p_2,
    w_m_1 + w_m_2,
)  # return: w+, w-

def Truth_revision(truth1: TruthV, truth2: TruthV):
    w_p_1 = fc_to_w_plus(truth1.f, truth1.c, truth1.k)
    w_p_2 = fc_to_w_plus(truth2.f, truth2.c, truth2.k)
    w_m_1 = fc_to_w_minus(truth1.f, truth1.c, truth1.k)
    w_m_2 = fc_to_w_minus(truth2.f, truth2.c, truth2.k)
    w_p, w_m = F_revision(w_p_1, w_p_2, w_m_1, w_m_2)
    truth = truth_from_w(w_p, w_m + w_p, truth1.k)
    return truth

class TruthV:
    k = 1

    def __init__(self, f: float, c: float):
        self.f = f
        self.c = c

    def revise(self, f: float, c: float, c_max=0.99):
        w_p_1 = fc_to_w_plus(self.f, self.c, TruthV.k)
        w_p_2 = fc_to_w_plus(f, c, TruthV.k)
        w_m_1 = fc_to_w_minus(self.f, self.c, TruthV.k)
        w_m_2 = fc_to_w_minus(f, c, TruthV.k)
        w_p, w_m = F_revision(w_p_1, w_p_2, w_m_1, w_m_2)
        self.f, self.c = w_to_f(w_p, w_m + w_p), w_to_c(w_m + w_p, TruthV.k)
        self.c = min(self.c, c_max)

    @property
    def e(self):
        return self.c * (self.f - 0.5) + 0.5

    def choose(self, other: TruthV):
        if TruthV.sharpness(self) < TruthV.sharpness(other):
            self.f, self.c = other.f, other.c

    @staticmethod
    def sharpness(truthv: TruthV) -> float:
        return abs(truthv.e - 0.5) * 2

    def __repr__(self):
        return f"%{self.f:.2f},{self.c:.2f}%"


DesireV = TruthV


def truth_from_w(w_plus, w, k=1):
    f, c = (w_to_f(w_plus, w), w_to_c(w, k)) if w != 0 else (0.5, 0.0)
    return TruthV(f, c)


def w_from_truth(truth: TruthV):
    f, c, k = truth.f, truth.c, truth.k
    return fc_to_w_plus(f, c, k), fc_to_w_minus(f, c, k)


# F_ded
F_ded = F_deduction = lambda f1, c1, f2, c2: (
    And(f1, f2),
    And(f1, f2, c1, c2),
)  # return: f, c

Truth_deduction = lambda truth1, truth2: TruthV(
    *F_deduction(truth1.f, truth1.c, truth2.f, truth2.c)
)

Desire_deduction = lambda truthv, desirev: DesireV(
    truthv.f * desirev.f, truthv.c * desirev.c
)


# F_ind
F_ind = F_induction = lambda f1, c1, f2, c2: (
    And(f1, f2, c1, c2),
    And(f2, c1, c2),
)  # return: w+, w
Truth_induction = lambda truth1, truth2: truth_from_w(
    *F_induction(truth1.f, truth1.c, truth2.f, truth2.c), truth1.k
)
