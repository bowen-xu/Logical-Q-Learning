from weakref import WeakSet


from q_nal.concepts import Sequence, Concept, Schema


c2 = Concept(2)
s1 = Sequence(Concept(1), Concept(2))
s3 = Sequence(Concept(1), Concept(2))

sch1 = Schema(s1, c2)
sch2 = Schema(s3, c2)
sch3 = Schema(s3, c2)

seqs = WeakSet()
seqs.add(s1)
print(s1 in seqs)  # True
print(s3 in seqs)  # True, because s2 and s3 are equal
seqs.add(s3)
s = next(iter(seqs))
print(id(s))
print(id(s1))
print(id(s3))
