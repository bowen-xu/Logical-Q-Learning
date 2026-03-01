import unittest
import sys

sys.path.insert(0, "/Users/bowenxu/Codes/Logical-Q-Learning")

from lql.elements import Desire, Belief
from lql.nal import TruthV, DesireV


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

    def test_desire_choose_sharper(self):
        desire = Desire(DesireV(0.5, 0.5))
        sharper = DesireV(0.9, 0.9)
        desire.choose(sharper)
        assert desire.desirev.f == 0.9
        assert desire.desirev.c == 0.9

    def test_desire_choose_less_sharp(self):
        desire = Desire(DesireV(0.9, 0.9))
        less_sharp = DesireV(0.5, 0.5)
        desire.choose(less_sharp)
        assert desire.desirev.f == 0.9
        assert desire.desirev.c == 0.9

    def test_desire_overwrite(self):
        desire = Desire(DesireV(0.3, 0.3))
        new_dv = DesireV(0.7, 0.8)
        desire.overwrite(new_dv)
        assert desire.desirev.f == 0.7
        assert desire.desirev.c == 0.8


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

    def test_belief_eternal(self):
        belief = Belief()
        belief.eternal = True
        assert belief.eternal is True

    def test_belief_t_occur(self):
        belief = Belief()
        belief.t_occur = 100
        assert belief.t_occur == 100

    def test_belief_induction(self):
        belief = Belief()
        truth1 = TruthV(0.8, 0.6)
        truth2 = TruthV(0.7, 0.5)
        belief.induction(truth1, truth2)
        assert belief.truthv.f is not None
        assert belief.truthv.c is not None
