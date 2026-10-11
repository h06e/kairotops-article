# S8: true stop reason of each generation, read from the solver logs of the pipeline (OQ-53). The stored
# `iterations` / `converged` fields are those of the last *recorded* iteration (every 10, OQ-36), so they are stale.
#   python3 scripts/S8_stop_reasons.py <pipeline workdir>/runs  > docs/S8_report/stop_reasons.csv
#   python3 scripts/S8_stop_reasons.py <C-Kai workdir>/runs C-Kai > docs/S8_report/stop_reasons_ckai.csv   (OQ-55, with a fraction column)
import os, re, sys
rows = []
d = sys.argv[1]
fam = sys.argv[2] if len(sys.argv) > 2 else "O-Kai"
for fn in sorted(os.listdir(d), key=lambda s: (len(s), s)):
    if not fn.endswith(".log"):
        continue
    txt = open(os.path.join(d, fn)).read()
    m = re.search(r"\[S8 " + fam + r" l0=([0-9.]+) rf=([0-9.]+) f=([0-9.]+) .*?beta=([0-9.]+)", txt)
    if not m:
        continue
    l0, rf, f, beta = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
    its = re.findall(r"^\s*(\d+)\s+\|\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\|\s+(\S+)", txt, re.M)
    it, jbin, wbin = int(its[-1][0]), float(its[-1][4]), float(its[-1][5])
    ls = "line search failed" in txt
    stop = "line-search" if ls else ("tolerance" if jbin <= 1e-3 and beta >= 64 else "maxiter")
    rows.append((float(l0), float(rf), it, jbin, wbin, stop, f))
if fam == "O-Kai":
    print("l0,rf,iterations,J_Bin_final,w_Bin_final,stop")
    for r in sorted(rows, key=lambda r: (r[1], r[0])):
        print(f"{r[0]:g},{r[1]:g},{r[2]},{r[3]:.3e},{r[4]:.4g},{r[5]}")
else:
    print("fraction,l0,rf,iterations,J_Bin_final,w_Bin_final,stop")
    for r in sorted(rows, key=lambda r: (r[6], r[1], r[0])):
        print(f"{r[6]:g},{r[0]:g},{r[1]:g},{r[2]},{r[3]:.3e},{r[4]:.4g},{r[5]}")
