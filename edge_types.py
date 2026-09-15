import numpy as np
d = np.load("mb_circuit.npz")
pre, post = d["pre"], d["post"]
N = int(max(pre.max(), post.max())+1)
is_ALPN = np.zeros(N, bool); is_ALPN[d["inputs_ALPN"]] = True
is_KC = np.zeros(N, bool); is_KC[d["KC"]] = True
is_MBON = np.zeros(N, bool); is_MBON[d["MBON"]] = True
is_DAN = np.zeros(N, bool); is_DAN[d["DAN"]] = True
from collections import Counter
c = Counter()
for a, b in zip(pre, post):
    key = ("A" if is_ALPN[a] else "K" if is_KC[a] else "M" if is_MBON[a] else "D" if is_DAN[a] else "?") + "->" + ("A" if is_ALPN[b] else "K" if is_KC[b] else "M" if is_MBON[b] else "D" if is_DAN[b] else "?")
    c[key] += 1
print(c.most_common(20))
