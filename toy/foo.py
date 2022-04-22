import pickle
with open("rules", "rb") as f:
    l = pickle.load(f)

l1 = list()
for a in l:
    l2 = list()
    for b in a:
        if (len(b) == 4):
            l2.append(b)
    l1.append(l2)
with open("rules2", "wb") as f:
    pickle.dump(l1, f)