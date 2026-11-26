"""Stage 4: capacitance extraction from the layout, and tuning of the layout to the Stage 2 targets.

Built-in quasi-static solver (method of moments): zero-thickness conductors lying on the surface of an
infinite dielectric half-space (silicon). For a charge on that interface the Green's function is exactly
1/(4 pi eps0 eps_eff r) with eps_eff = (1 + eps_r)/2, so no volume mesh is needed. Cells form a graded
quadtree (small near conductor edges, large far away). Validated against the analytic capacitance of a
square plate. It ignores finite substrate thickness, the package and metal thickness, and has roughly
5 % accuracy. It is NOT a replacement for Palace or Elmer at sign-off; it makes the tuning loop run
anywhere in minutes.

Steps: (1) validate, (2) extract capacitances in windows around each element, (3) tune island arm length,
coupling-head width and resonator coupling length until the capacitances match Stage 2, (4) correct the
resonator lengths for loading, (5) re-run the Stage 3 design-rule checks and write the tuned GDS files.
Run: python stage4_extract.py   (needs stage2_targets.json from stage2_quantization.py)
"""
import functools
import json
import math
import sys
import time
import numpy as np
import shapely
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter
from scipy.optimize import brentq
from scipy.special import ellipk
from shapely.geometry import box
import chipset as cs
import stage3_layout as s3
from chipset import B, ok

print = functools.partial(print, flush=True)
EPS0 = 8.8541878128e-12
C_LIGHT = 2.99792458e8
EPS_EFF = B("eps_eff")
PITCH, GRAD, SMAX = 3.0, 1.5, 32          # raster pitch (um), grading factor, largest cell (pixels)
MARK = lambda c: "PASS" if c else "NOT MET"


# ---------------------------------------------------------------- solver
def quadtree(lab, dist, pitch, grad, smax=16):
    NY, NX = lab.shape
    assigned = np.zeros(lab.shape, bool)
    xs, ys, sz, lb = [], [], [], []
    s = smax
    while s >= 1:
        by, bx = NY // s, NX // s
        L4 = lab.reshape(by, s, bx, s)
        mn, mx = L4.min(axis=(1, 3)), L4.max(axis=(1, 3))
        d4 = dist.reshape(by, s, bx, s).min(axis=(1, 3))
        a4 = assigned.reshape(by, s, bx, s).any(axis=(1, 3))
        good = (mn == mx) & (mn > 0) & (~a4) & ((s <= 1) | (s <= grad * d4 + 1))
        iy, ix = np.nonzero(good)
        xs.append((ix * s + s / 2) * pitch); ys.append((iy * s + s / 2) * pitch)
        sz.append(np.full(len(ix), s * pitch)); lb.append(mn[iy, ix])
        A = assigned.reshape(by, s, bx, s); A[iy, :, ix, :] = True     # mark blocks (view into assigned)
        s //= 2
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(sz), np.concatenate(lb).astype(int)


def maxwell(xc, yc, sz, lab, M, eps_eff, block=512):
    """Maxwell capacitance matrix in fF for M conductors (labels 1..M)."""
    N = len(xc)
    P = np.empty((N, N))
    off = (np.arange(4) + 0.5) / 4 - 0.5
    ox, oy = (v.ravel() for v in np.meshgrid(off, off))
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        r = np.hypot(xc[i0:i1, None] - xc[None, :], yc[i0:i1, None] - yc[None, :])
        with np.errstate(divide="ignore"):
            Pb = 1.0 / r
        ii, jj = np.nonzero(r < 2.5 * sz[None, :])
        gi = ii + i0; keep = gi != jj
        ii, jj, gi = ii[keep], jj[keep], gi[keep]
        if len(ii):
            sx = xc[jj][:, None] + ox[None, :] * sz[jj][:, None]
            sy = yc[jj][:, None] + oy[None, :] * sz[jj][:, None]
            Pb[ii, jj] = (1.0 / np.hypot(xc[gi][:, None] - sx, yc[gi][:, None] - sy)).mean(axis=1)
        P[i0:i1] = Pb
    P[np.arange(N), np.arange(N)] = 4 * math.log(1 + math.sqrt(2)) / sz       # uniform square, centre point
    P *= 1e6 / (4 * math.pi * EPS0 * eps_eff)                                # um -> m
    P = 0.5 * (P + P.T)
    A = np.zeros((N, M)); A[np.arange(N), lab - 1] = 1.0
    try:
        X = cho_solve(cho_factor(P), A)
    except Exception:
        X = solve(P, A)
    C = A.T @ X
    return 0.5 * (C + C.T) * 1e15


def solve_window(window, conductors, pitch=PITCH, grad=GRAD, smax=SMAX):
    """conductors: ordered list of (name, shapely geometry or None). Returns (names, C[fF], n_cells)."""
    x0, y0, x1, y1 = window.bounds
    nx, ny = int(round((x1 - x0) / pitch)), int(round((y1 - y0) / pitch))
    assert nx % 16 == 0 and ny % 16 == 0
    X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5) * pitch, y0 + (np.arange(ny) + 0.5) * pitch)
    lab = np.zeros((ny, nx), np.int16)
    live = []
    for name, g in conductors:
        if g is None or g.is_empty:
            continue
        live.append(name)
        m = shapely.contains_xy(g, X.ravel(), Y.ravel()).reshape(ny, nx)
        lab[m] = len(live)
    mx_ = maximum_filter(lab, size=3, mode="nearest"); mn_ = minimum_filter(lab, size=3, mode="nearest")
    interior = mx_ == mn_
    dist = distance_transform_edt(interior) if (~interior).any() else np.full(lab.shape, 1e6)
    xc, yc, sz, lb = quadtree(lab, dist, pitch, grad, smax)
    xc, yc = xc + x0, yc + y0
    return live, maxwell(xc, yc, sz, lb, len(live), EPS_EFF), len(xc)


def make_window(cx, cy, w, h, pitch=PITCH):
    q = SMAX * pitch
    w, h = math.ceil(w / q) * q, math.ceil(h / q) * q
    return box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def layout_conductors(L, win):
    """Group layout nets into conductors: islands+wires share a name, shorted nets are ground."""
    conds = {"GND": [L.ground.intersection(win)]}
    for net, geom in L.net_u.items():
        g = geom.intersection(win)
        if g.is_empty:
            continue
        k = s3.kind(net)
        name = "GND" if (net in L.shorted and k == "line") else (s3.group(net) if (k == "island" or k.startswith("wire")) else net)
        conds.setdefault(name, []).append(g)
    return [(n, shapely.union_all(v)) for n, v in conds.items()]


def extract(L, win, pitch=PITCH):
    names, C, n = solve_window(win, layout_conductors(L, win), pitch)
    return {"names": names, "C": C, "cells": n}


# ---------------------------------------------------------------- validation
def plate_test(pitch, a=100.):
    win = make_window(0, 0, 400, 400, pitch)
    X = box(-a / 2, -a / 2, a / 2, a / 2)
    names, C, n = solve_window(win, [("plate", X)], pitch)
    want = 0.3667 * 4 * math.pi * EPS0 * a * 1e-6 * EPS_EFF * 1e15     # fF; 0.367 from numerical studies (older value 0.3607)
    return C[0, 0], want, n


# ---------------------------------------------------------------- metrics for each die
def cell(r, a, b=None):
    i = r["names"].index(a)
    return r["C"][i, i] if b is None else -r["C"][i, r["names"].index(b)]


def metrics_K(L, pitch=PITCH):
    m = {}
    r = extract(L, make_window(4200, 5200, 1344, 1216, pitch), pitch)
    m["Csig_K0"], m["C_qr_K0"] = cell(r, "K0"), cell(r, "K0", "K0:res")
    r = extract(L, make_window(5800, 5200, 1344, 1216, pitch), pitch)
    m["Csig_K1"], m["C_qr_K1"] = cell(r, "K1"), cell(r, "K1", "K1:res")
    r = extract(L, make_window(5000, 5200, 1344, 1216, pitch), pitch)
    m["Csig_CP"], m["C_qq"] = cell(r, "CP"), cell(r, "K0", "K1")
    m["C_qc_K0"], m["C_qc_K1"] = cell(r, "CP", "K0"), cell(r, "CP", "K1")
    m["cells"] = r["cells"]
    return m


def metrics_H(L, pitch=PITCH):
    m = {}
    for n, cx in (("H0", 3800), ("H1", 6200)):
        r = extract(L, make_window(cx, 4650, 1280, 1152, pitch), pitch)
        m[f"Csig_{n}"], m[f"C_qr_{n}"] = cell(r, n), cell(r, n, f"{n}:res")
    r = extract(L, make_window(5000, 5950, 1536, 1152, pitch), pitch)
    m["Csig_H2"], m["C_qr_H2"] = cell(r, "H2"), cell(r, "H2", "H2:res")
    m["C_e02"], m["C_e12"] = cell(r, "H2", "H0"), cell(r, "H2", "H1")
    m["C_e01"] = cell(extract(L, make_window(5500, 4600, 512, 448, pitch), pitch), "H0", "H1")
    m["cells"] = r["cells"]
    return m


# ---------------------------------------------------------------- tuning
def tune(build, metrics_fn, specs, label, iters=9, tol=0.02, delta=0.15):
    """specs: {param: (metric names, target, (lo, hi), start)}; each parameter drives the mean of its metrics."""
    names = list(specs)
    def ev(p):
        L = build(p); L.build()
        m = metrics_fn(L)
        return {k: float(np.mean([m[x] for x in specs[k][0]])) for k in names}, m
    p = {k: specs[k][3] for k in names}
    got, raw = ev(p)
    hist = [(dict(p), dict(got))]
    q = {k: float(np.clip(p[k] * (1 + delta), *specs[k][2])) for k in names}
    got2, raw = ev(q); hist.append((dict(q), dict(got2)))
    sens = {k: (got2[k] - got[k]) / (q[k] - p[k]) if abs(q[k] - p[k]) > 1e-9 else 1.0 for k in names}
    p, got = q, got2
    for it in range(iters):
        err = {k: (specs[k][1] - got[k]) / specs[k][1] for k in names}
        print(f"  {label} iteration {it}: " + ", ".join(f"{k}={p[k]:.1f} ({100 * -err[k]:+.1f}%)" for k in names))
        stuck = lambda k: (p[k] <= specs[k][2][0] + 1e-6 and err[k] < 0) or (p[k] >= specs[k][2][1] - 1e-6 and err[k] > 0)
        if all(abs(e) < tol or stuck(k) for k, e in err.items()):
            break
        new = {}
        for k in names:
            lo, hi = specs[k][2]
            new[k] = float(np.clip(p[k] + 0.9 * (specs[k][1] - got[k]) / (sens[k] if abs(sens[k]) > 1e-9 else 1.0), lo, hi))
        g_new, raw = ev(new)
        for k in names:
            if abs(new[k] - p[k]) > 1e-3 * p[k]:
                s_ = (g_new[k] - got[k]) / (new[k] - p[k])
                if s_ * sens[k] > 0:
                    sens[k] = s_
        p, got = new, g_new
        hist.append((dict(p), dict(got)))
    return p, got, raw, hist


# ---------------------------------------------------------------- resonator loading
def cpw_line(w=10., g=6.):
    k = w / (w + 2 * g)
    K, Kp = ellipk(k ** 2), ellipk(1 - k ** 2)
    Z0 = 30 * math.pi / math.sqrt(EPS_EFF) * Kp / K
    v = C_LIGHT / math.sqrt(EPS_EFF)
    return Z0, v, math.sqrt(EPS_EFF) / (C_LIGHT * Z0)              # ohm, m/s, F/m


def f_loaded(length_um, CL_fF, Z0, v):
    """Quarter-wave line, shorted at one end, open end loaded by CL: tan(beta l) * w CL Z0 = 1."""
    l = length_um * 1e-6; f0 = v / (4 * l)
    if CL_fF <= 0:
        return f0
    g = lambda f: math.tan(2 * math.pi * f * l / v) * 2 * math.pi * f * CL_fF * 1e-15 * Z0 - 1
    return brentq(g, 1e-3 * f0, f0 * (1 - 1e-9))


def corrected_length(f_target_GHz, len0_um, CL_fF, Z0, v):
    return brentq(lambda l: f_loaded(l, CL_fF, Z0, v) - f_target_GHz * 1e9, 0.8 * len0_um, 1.05 * len0_um)


# ---------------------------------------------------------------- main
def checks_for(prefix, final, targets, tolfrac=0.05):
    rows = []
    for k, t in targets.items():
        val = final[k]; rel = (val - t) / t
        rows.append({"metric": k, "target_fF": t, "extracted_fF": float(val), "error_pct": float(100 * rel), "pass": abs(rel) < tolfrac})
        print(f" {prefix} {k:<9} target {t:7.3f} fF  extracted {val:7.3f} fF  ({100 * rel:+.1f}%): {ok(abs(rel) < tolfrac)}")
    return rows


def status_rows(prefix, vals, targets, tolfrac=0.05):
    rows = []
    for k, t in targets.items():
        rel = (vals[k] - t) / t
        rows.append({"metric": k, "target_fF": t, "extracted_fF": float(vals[k]), "error_pct": float(100 * rel), "met": abs(rel) < tolfrac})
        print(f" {prefix} {k:<9} target {t:7.3f} fF  extracted {vals[k]:8.3f} fF  ({100 * rel:+7.1f}%): {MARK(abs(rel) < tolfrac)}")
    return rows


def conductor_length(L, group_name):
    """Approximate attached conductor length (um): arms plus wires, from polygon area / width."""
    tot = 0.0
    for n, g in L.net_u.items():
        if s3.group(n) == group_name and (s3.kind(n) == "island" or s3.kind(n).startswith("wire")):
            tot += g.area / (s3.R["arm_w"] if s3.kind(n) == "island" else s3.R["wire_w"])
    return tot


def main(which):
    P = s3.fmt_targets()
    Z0, v, Cl = cpw_line()
    tr, cc, rs = P["transmons"], P["capacitance"]["coupling_fF"], P["capacitance"]["resonators"]
    CsK, CsH, CsC = tr["kahler_die"]["C_sigma_fF"], tr["hyperkahler_die"]["C_sigma_fF"], B("coupler_C_fF")
    print(f"CPW 10/6 on silicon: Z0 = {Z0:.1f} ohm, v = {v / 1e8:.3f}e8 m/s, C' = {Cl * 1e9:.3f} fF/um")
    print("Solver validation (100 um square plate on silicon, reference 0.367 x 4 pi eps0 a x eps_eff)")
    c_f, want, n_f = plate_test(1.5); c_p, _, n_p = plate_test(PITCH)
    print(f" reference {want:.2f} fF | pitch 1.5 um {c_f:.2f} fF ({100 * (c_f / want - 1):+.1f}%, {n_f} cells) | "
          f"production pitch {PITCH:g} um {c_p:.2f} fF ({100 * (c_p / want - 1):+.1f}%, {n_p} cells)")
    print(f" fine mesh within 2% and production mesh within 3% of the reference: {ok(abs(c_f / want - 1) < 0.02 and abs(c_p / want - 1) < 0.03)}")
    print("Capacitance budget")
    Lc = s3.Layout("probe"); Lc.cross("X:island", 5000, 5000, 210.); Lc.build()
    Ccross = cell(extract(Lc, make_window(5000, 5000, 960, 960)), "X")
    print(f" island alone (cross, arms 210 um, 24 um wide): {Ccross:.1f} fF = {Ccross / (4 * 210):.3f} fF per um of arm; "
          f"a bare 8 um coupling wire adds about 0.105 fF per um")

    dies = {
        "K": dict(name="Kähler die", tag="kahler", build=lambda t: s3.build_kahler(P, t), mfn=metrics_K, groups=["K0", "CP", "K1"],
                  specs={"head_qc": (["C_qc_K0", "C_qc_K1"], cc["kahler_qc"], (6., 150.), cc["kahler_qc"] / s3.CPL),
                         "head_qq": (["C_qq"], cc["kahler_qq"], (6., 100.), 8.),
                         "lq_K0": (["C_qr_K0"], rs["K0"]["C_qr_fF"], (10., 190.), rs["K0"]["C_qr_fF"] / s3.CPL),
                         "lq_K1": (["C_qr_K1"], rs["K1"]["C_qr_fF"], (10., 190.), rs["K1"]["C_qr_fF"] / s3.CPL)},
                  targets={"C_qc_K0": cc["kahler_qc"], "C_qc_K1": cc["kahler_qc"], "C_qq": cc["kahler_qq"], "C_qr_K0": rs["K0"]["C_qr_fF"],
                           "C_qr_K1": rs["K1"]["C_qr_fF"], "Csig_K0": CsK, "Csig_K1": CsK, "Csig_CP": CsC},
                  zoom=((3300, 6700), (4300, 6800))),
        "H": dict(name="Hyperkähler die", tag="hyperkahler", build=lambda t: s3.build_hyper(P, t), mfn=metrics_H, groups=["H0", "H1", "H2"],
                  specs={"head_h": (["C_e01", "C_e02", "C_e12"], cc["hyperkahler_edge"], (6., 100.), 8.),
                         "lq_H0": (["C_qr_H0"], rs["H0"]["C_qr_fF"], (10., 190.), rs["H0"]["C_qr_fF"] / s3.CPL),
                         "lq_H1": (["C_qr_H1"], rs["H1"]["C_qr_fF"], (10., 190.), rs["H1"]["C_qr_fF"] / s3.CPL),
                         "lq_H2": (["C_qr_H2"], rs["H2"]["C_qr_fF"], (10., 190.), rs["H2"]["C_qr_fF"] / s3.CPL)},
                  targets={"C_e01": cc["hyperkahler_edge"], "C_e02": cc["hyperkahler_edge"], "C_e12": cc["hyperkahler_edge"],
                           "C_qr_H0": rs["H0"]["C_qr_fF"], "C_qr_H1": rs["H1"]["C_qr_fF"], "C_qr_H2": rs["H2"]["C_qr_fF"],
                           "Csig_H0": CsH, "Csig_H1": CsH, "Csig_H2": CsH},
                  zoom=((2800, 7300), (4000, 6500)))}
    for key in which:
        d = dies[key]; t0 = time.time()
        print(f"{d['name']}: extraction of the Stage 3 layout, then tuning of coupling capacitors and resonator couplings")
        L0 = d["build"]({}); L0.build(); m0 = d["mfn"](L0)
        for g_ in d["groups"]:
            print(f"  {g_}: {conductor_length(L0, g_):.0f} um of attached arm and wire conductor")
        p, got, raw, hist = tune(d["build"], d["mfn"], d["specs"], d["name"], iters=5, tol=0.03)
        corr = {}
        for n in [k.split("_")[1] for k in d["specs"] if k.startswith("lq_")]:
            Cq, Cs = raw[f"C_qr_{n}"], raw[f"Csig_{n}"]
            CL = Cq * (Cs - Cq) / Cs
            len0, ft = rs[n]["length_mm"] * 1e3, rs[n]["f_res_GHz"]
            fl = f_loaded(len0, CL, Z0, v) / 1e9
            ln = corrected_length(ft, len0, CL, Z0, v)
            corr[n] = {"C_qr_fF": Cq, "C_load_fF": CL, "f_loaded_at_stage2_length_GHz": fl, "target_GHz": ft,
                       "length_stage2_um": len0, "length_corrected_um": ln, "scale": ln / len0}
            print(f"  {n} resonator: loaded by {CL:.2f} fF -> {fl:.3f} GHz at the Stage 2 length (target {ft:.2f}); length {len0:.0f} -> {ln:.0f} um ({100 * (ln / len0 - 1):+.2f}%)")
        tf = dict(p); tf.update({f"len_{n}": c["scale"] for n, c in corr.items()})
        L1 = d["build"](tf); L1.build(); mf = d["mfn"](L1)
        rows = status_rows(key, mf, d["targets"])
        L1.stitch(); L1.build()
        res = s3.drc(L1, None); bad = [r for r in res if not r["pass"]]
        print(f" {d['tag']} die, {len(res)} design-rule checks on the tuned layout: {ok(not bad)}" + (f"  failed: {[r['check'] for r in bad]}" if bad else ""))
        s3.to_gds(L1, f"{d['tag']}_die_stage4.gds"); s3.draw(L1, f"{d['tag']}_die_stage4.png"); s3.draw(L1, f"{d['tag']}_die_stage4_zoom.png", *d["zoom"])
        coup_ok = all(r["met"] for r in rows if not r["metric"].startswith("Csig"))
        csig_ok = all(r["met"] for r in rows if r["metric"].startswith("Csig"))
        print(f" {d['name']}: coupling capacitors and resonator couplings within 5% of Stage 2: {MARK(coup_ok)}")
        print(f" {d['name']}: island total capacitance within 5% of Stage 2: {MARK(csig_ok)}")
        print(f" {d['name']}: resonator length corrections all below 3%: {ok(max(abs(c['scale'] - 1) for c in corr.values()) < 0.03)}  ({time.time() - t0:.0f} s)")
        json.dump({"tune": tf, "untuned": {k: float(x) for k, x in m0.items()}, "tuned": {k: float(x) for k, x in mf.items()}, "targets": rows,
                   "resonators": corr, "drc_pass": not bad, "n_checks": len(res), "validation_fF": [want, c_f, c_p], "cross_arm210_fF": Ccross,
                   "solver": {"eps_eff": EPS_EFF, "pitch_um": PITCH, "grading": GRAD, "note": "quasi-static MoM, zero-thickness, infinite substrate, about 5% accuracy"}},
                  open(f"stage4_{key}.json", "w"), indent=2, default=float)
    done = [k for k in "KH" if __import__("os").path.exists(f"stage4_{k}.json")]
    json.dump({k: json.load(open(f"stage4_{k}.json")) for k in done}, open("stage4_extracted.json", "w"), indent=2)
    if len(done) == 2:
        worst = max(r["extracted_fF"] / r["target_fF"] for k in done for r in json.load(open(f"stage4_{k}.json"))["targets"] if r["metric"].startswith("Csig"))
        if worst > 1.05:
            print(f"FINDING: the Stage 3 floorplan is over its capacitance budget by up to {worst:.1f}x, because the coupling wires are long.")
            print(" It needs a compact re-layout (each qubit has roughly 880 um of conductor in total), not more tuning.")
    print("wrote stage4_extracted.json" + ("" if len(done) == 2 else f" (dies done so far: {', '.join(done)})"))


if __name__ == "__main__":
    main([c for c in (sys.argv[1] if len(sys.argv) > 1 else "KH")])
