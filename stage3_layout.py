"""Stage 3: parametric layout (GDS) for both dies, with design-rule checks under ASSUMED rules.

Reads the Stage 2 targets (stage2_targets.json), draws each die, checks it, writes
  kahler_die.gds, hyperkahler_die.gds, previews (PNG) and stage3_layout_report.json.

Everything here is a first-pass floorplan: island sizes, coupling lengths and resonator/filter
coupling lengths use rule-of-thumb capacitance per length (CPL) and are TUNED IN STAGE 4 (EM).
The design rules below are generic, not a foundry's. Junction geometry is an idealised overlap.
Run: python stage3_layout.py
"""
import json
import math
import numpy as np
import gdstk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.optimize import brentq
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union
import chipset as cs
import stage2_quantization as q2
from chipset import B, ok

R = dict(cpw_w=10., cpw_g=6., bend_r=50., min_metal=4., min_gap=4., net_space=8., island_clear=30.,
         wire_w=8., wire_clear=22., arm=210., arm_w=24., edge_ground_margin=50., edge_keepout=150.,
         strap_pitch=700., strap_span_w=10., strap_pad=24., strap_max_span=150., strap_min_gap=15.,
         island_min_area=500., jj_min_um=0.10, jj_max_um=0.40, launcher_pitch=500., s_g=6.)
CPL = 0.08            # ASSUMED fF per um of facing edge for ~10 um gaps (EM-tuned in Stage 4)
W = H = 10000.
Y_R = 8700. - (R["cpw_w"] + 2 * R["cpw_g"] + R["s_g"])      # resonator coupling-run centreline (8672)


# ---------------------------------------------------------------- geometry helpers
def unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def rect(c, u, length, width):
    c = np.array(c, float); u = unit(u); n = np.array([-u[1], u[0]])
    a, b = u * length / 2, n * width / 2
    return Polygon([c - a - b, c + a - b, c + a + b, c - a + b])


def rounded(pts, r, n=16):
    """Polyline with circular arcs of radius r (clamped) at every corner."""
    P = [np.array(p, float) for p in pts]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        p0, p1, p2 = P[i - 1], P[i], P[i + 1]
        u1 = unit(p1 - p0); u2 = unit(p2 - p1)
        l1, l2 = np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1)
        ang = math.acos(np.clip(u1 @ u2, -1, 1))
        if ang < 1e-9:
            out.append(p1); continue
        t = min(r * math.tan(ang / 2), l1 / 2, l2 / 2)
        rr = t / math.tan(ang / 2)
        a = p1 - u1 * t
        sgn = 1 if (u1[0] * u2[1] - u1[1] * u2[0]) > 0 else -1
        c = a + np.array([-u1[1], u1[0]]) * sgn * rr
        v0 = a - c
        for k in range(n + 1):
            th = sgn * ang * k / n
            out.append(c + np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) @ v0)
    out.append(P[-1])
    return np.array(out)


def plen(sp):
    return float(np.sum(np.linalg.norm(np.diff(sp, axis=0), axis=1)))


def min_radius(pts, r):
    """Smallest bend radius actually achieved after clamping."""
    m = r
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = (np.array(p, float) for p in pts[i - 1:i + 2])
        u1, u2 = unit(p1 - p0), unit(p2 - p1)
        ang = math.acos(np.clip(u1 @ u2, -1, 1))
        if ang < 1e-9:
            continue
        t = min(r * math.tan(ang / 2), np.linalg.norm(p1 - p0) / 2, np.linalg.norm(p2 - p1) / 2)
        m = min(m, t / math.tan(ang / 2))
    return m


def group(net):
    return net.split(":")[0]


def kind(net):
    return net.split(":")[1]


# ---------------------------------------------------------------- layout container
class Layout:
    def __init__(s, name):
        s.name = name
        s.etch, s.nets, s.jj1, s.jj2, s.straps = [], {}, [], [], []
        s.junctions, s.ports, s.shorted, s.lengths, s.min_r = [], [], set(), {}, 1e9
        s.stitch_failed, s.heads = [], []

    def add(s, net, poly):
        s.nets.setdefault(net, []).append(poly)

    def cpw(s, pts, net, start="free", end="free", r=None, w=None, g=None):
        r, w, g = r or R["bend_r"], w or R["cpw_w"], g or R["cpw_g"]
        sp = rounded(pts, r)
        s.min_r = min(s.min_r, min_radius(pts, r))
        ext = lambda a, b, d: a + unit(a - b) * d
        es, ce = [tuple(p) for p in sp], [tuple(p) for p in sp]
        if start == "open": es.insert(0, tuple(ext(sp[0], sp[1], g)))
        if end == "open": es.append(tuple(ext(sp[-1], sp[-2], g)))
        if start == "short": ce.insert(0, tuple(ext(sp[0], sp[1], 8.)))
        if end == "short": ce.append(tuple(ext(sp[-1], sp[-2], 8.)))
        s.etch.append(LineString(es).buffer(w / 2 + g, cap_style=2, join_style=1))
        s.add(net, LineString(ce).buffer(w / 2, cap_style=2, join_style=1))
        if "short" in (start, end): s.shorted.add(net)
        return sp

    def cross(s, net, cx, cy, arm=None):
        a, w = (arm or R["arm"]), R["arm_w"]
        p = unary_union([box(cx - a, cy - w / 2, cx + a, cy + w / 2), box(cx - w / 2, cy - a, cx + w / 2, cy + a)])
        s.add(net, p); s.etch.append(p.buffer(R["island_clear"], join_style=2))

    def wire(s, pts, net, head=8., r=30.):
        sp = rounded(pts, r)
        ls = LineString(sp)
        u = unit(sp[-1] - sp[-2])
        hd = rect(sp[-1] + u * 4, u, 8., head)
        s.add(net, ls.buffer(R["wire_w"] / 2, cap_style=2, join_style=1)); s.add(net, hd); s.heads.append(hd)
        s.etch.append(unary_union([ls.buffer(R["wire_w"] / 2 + R["wire_clear"], cap_style=2, join_style=1),
                                   hd.buffer(R["wire_clear"], join_style=2)]))
        return sp

    def squid(s, x, y_tip, s1, s2, label):
        """SQUID in the 30 um gap below an arm tip: bottom bar in ground (jj1), top bar on the arm (jj2)."""
        yj = y_tip - 16
        for xc, sd in ((x - 8, s1), (x + 8, s2)):
            s.jj1.append(box(xc - sd / 2, y_tip - 36, xc + sd / 2, yj + sd))
            s.jj2.append(box(xc - sd / 2, yj, xc + sd / 2, y_tip + 4))
        s.jj1.append(box(x - 8 - s1 / 2, y_tip - 36, x + 8 + s2 / 2, y_tip - 28))
        s.jj2.append(box(x - 8 - s1 / 2, y_tip - 2, x + 8 + s2 / 2, y_tip + 4))
        s.junctions.append(dict(label=label, side1_um=s1, side2_um=s2, x=x, y=yj))

    def launcher(s, net, x, y, d, name):
        u = unit(d); n = np.array([-u[1], u[0]]); o = np.array([x, y], float)
        P = lambda a, b: o + u * a + n * b
        s.add(net, Polygon([P(0, -100), P(0, 100), P(300, 100), P(600, 5), P(600, -5), P(300, -100)]))
        s.etch.append(Polygon([P(-5, -200), P(-5, 200), P(300, 200), P(600, 11), P(600, -11), P(300, -200)]))
        s.ports.append(dict(name=name, x=float(x), y=float(y)))
        return P(600, 0)

    def strap(s, c, u, half_etch, kind_="line"):
        u = unit(u); d = half_etch + 16
        c = np.array(c, float)
        s.straps.append(dict(p1=rect(c - u * d, u, R["strap_pad"], R["strap_pad"]),
                             p2=rect(c + u * d, u, R["strap_pad"], R["strap_pad"]),
                             span=rect(c, u, 2 * d + R["strap_pad"], R["strap_span_w"]), kind=kind_))

    def strap_line(s, wp, half_etch=11., margin=250.):
        for a, b in zip(wp[:-1], wp[1:]):
            a, b = np.array(a, float), np.array(b, float)
            L = np.linalg.norm(b - a)
            if L < 2 * margin + 100:
                continue
            u = unit(b - a); n = np.array([-u[1], u[0]])
            k = int((L - 2 * margin) // R["strap_pitch"]) + 1
            for t in margin + (L - 2 * margin) * (np.arange(k) + 0.5) / k:
                s.strap(a + u * t, n, half_etch)

    def crossover(s, a, b, wire_half=4.):
        """Signal airbridge: spans from landing block a to landing block b (both on the same wire net)."""
        a, b = np.array(a, float), np.array(b, float); u = unit(b - a)
        s.straps.append(dict(p1=rect(a, u, 36, 36), p2=rect(b, u, 36, 36), span=rect((a + b) / 2, u, np.linalg.norm(b - a) + 24, 12), kind="crossover"))

    # ------------------------------------------------------------ build and check
    def build(s):
        s.inner = box(R["edge_ground_margin"], R["edge_ground_margin"], W - R["edge_ground_margin"], H - R["edge_ground_margin"])
        s.etch_u = unary_union(s.etch)
        s.ground = s.inner.difference(s.etch_u)
        s.net_u = {n: unary_union(p) for n, p in s.nets.items()}
        s.metal = unary_union([s.ground] + list(s.net_u.values()))
        s.sensitive = unary_union([v.buffer(100) for n, v in s.net_u.items() if kind(n) == "island"]
                                  + [g.buffer(60) for g in s.jj1 + s.jj2] + [g.buffer(120) for g in s.heads])

    def comps(s):
        g = s.ground
        return list(g.geoms) if hasattr(g, "geoms") else [g]

    def stitch(s):
        """Add ground straps until every ground region is connected to the main plane."""
        comps = s.comps(); parent = list(range(len(comps)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i

        def hits(poly):
            return [i for i, c in enumerate(comps) if c.intersection(poly).area > 50]

        def link(st):
            h = list(set(hits(st["p1"]) + hits(st["p2"])))
            for a in h[1:]:
                parent[find(a)] = find(h[0])
        for st in s.straps:
            if st["kind"] != "crossover":
                link(st)
        main = max(range(len(comps)), key=lambda i: comps[i].area)
        skip = set()
        for _ in range(80):
            iso = [i for i in range(len(comps)) if find(i) != find(main) and comps[i].area > R["island_min_area"] and i not in skip]
            if not iso:
                break
            i = max(iso, key=lambda k: comps[k].area)
            other = unary_union([comps[k] for k in range(len(comps)) if find(k) == find(main)])
            bd = comps[i].boundary; best = None
            for t in np.arange(0, bd.length, 20.):
                p = bd.interpolate(t); q = nearest_points(p, other)[1]; d = p.distance(q)
                if not (R["strap_min_gap"] <= d <= R["strap_max_span"] - 54):
                    continue
                n = unit([q.x - p.x, q.y - p.y])
                c1, c2 = np.array([p.x, p.y]) - n * 15, np.array([q.x, q.y]) + n * 15
                p1, p2 = rect(c1, n, 24, 24), rect(c2, n, 24, 24)
                if not (p1.within(comps[i]) and p2.within(other)):
                    continue
                span = rect((c1 + c2) / 2, n, np.linalg.norm(c2 - c1) + 24, 10)
                if span.intersects(s.sensitive) or any(span.distance(x["span"]) < 30 for x in s.straps):
                    continue
                if best is None or d < best[0]:
                    best = (d, p1, p2, span)
            if best is None:
                skip.add(i); s.stitch_failed.append(float(comps[i].area)); continue
            s.straps.append(dict(p1=best[1], p2=best[2], span=best[3], kind="auto"))
            link(s.straps[-1])
        s.pre_islands = len([c for c in comps if c.area > R["island_min_area"]]) - 1
        s.parent_roots = len({find(i) for i in range(len(comps)) if comps[i].area > R["island_min_area"]})
        s.n_comps = len([c for c in comps if c.area > R["island_min_area"]])
        s.main_area = comps[main].area


# ---------------------------------------------------------------- DRC
def thin(geom, w):
    return geom.difference(geom.buffer(-w / 2, join_style=2).buffer(w / 2, join_style=2))


def drc(L, targets):
    out = []
    chk = lambda name, cond, detail="": out.append({"check": name, "pass": bool(cond), "detail": detail})
    # widths
    tm = thin(L.metal.intersection(L.inner), R["min_metal"])
    chk("min metal width >= %g um" % R["min_metal"], tm.area < 20, f"thin area {tm.area:.1f} um2")
    tg = thin(L.inner.difference(L.metal), R["min_gap"])
    chk("min gap width >= %g um" % R["min_gap"], tg.area < 20, f"thin area {tg.area:.1f} um2")
    # net spacing and unintended grounding
    names = list(L.net_u)
    worst = 1e9; viol = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if group(names[a]) == group(names[b]):
                continue
            d = L.net_u[names[a]].distance(L.net_u[names[b]])
            worst = min(worst, d)
            if d < R["net_space"]:
                viol.append((names[a], names[b], round(d, 1)))
    chk("net-to-net spacing >= %g um" % R["net_space"], not viol, f"closest {worst:.1f} um; {viol[:4]}")
    bad = [n for n in names if n not in L.shorted and kind(n) != "launcher" and L.net_u[n].intersection(L.ground).area > 0.5]
    chk("no unintended contact between signal nets and ground", not bad, str(bad[:5]))
    # connectivity
    chk("ground plane is one connected region (after straps)", L.parent_roots == 1 and not L.stitch_failed,
        f"{L.n_comps} regions >{R['island_min_area']:.0f} um2, {len(L.straps)} straps, unresolved {L.stitch_failed}")
    pads_ok = all(st["p1"].within(L.ground.buffer(0.01)) and st["p2"].within(L.ground.buffer(0.01))
                  for st in L.straps if st["kind"] != "crossover")
    bad_pads = sum(1 for st in L.straps if st["kind"] != "crossover" and not (st["p1"].within(L.ground.buffer(0.01)) and st["p2"].within(L.ground.buffer(0.01))))
    chk("every airbridge landing pad sits fully on ground", pads_ok, f"{bad_pads} pads off ground")
    spans = [math.dist(st["span"].bounds[:2], st["span"].bounds[2:]) for st in L.straps]
    chk("airbridge span <= %g um" % R["strap_max_span"],
        all(max(st["span"].bounds[2] - st["span"].bounds[0], st["span"].bounds[3] - st["span"].bounds[1]) <= R["strap_max_span"] + 25 for st in L.straps),
        f"{len(L.straps)} bridges")
    # keep-out and bends
    ring = box(0, 0, W, H).difference(box(R["edge_keepout"], R["edge_keepout"], W - R["edge_keepout"], H - R["edge_keepout"]))
    intr = [n for n in names if kind(n) in ("island", "wire", "res") and L.net_u[n].intersects(ring)]
    chk("islands, wires and resonators stay inside the %g um edge keep-out" % R["edge_keepout"], not intr, str(intr))
    chk("minimum CPW bend radius >= 40 um", L.min_r >= 40, f"{L.min_r:.1f} um")
    # junctions
    jr, areas_ok = True, True
    for j in L.junctions:
        jr &= all(R["jj_min_um"] <= j[k] <= R["jj_max_um"] for k in ("side1_um", "side2_um"))
        near = lambda g: abs(g.centroid.x - j["x"]) < 40 and abs(g.centroid.y - j["y"]) < 40
        cells = sorted(a.intersection(b).area for a in L.jj1 if near(a) for b in L.jj2 if near(b) and a.intersects(b))
        want = sorted([j["side1_um"] ** 2, j["side2_um"] ** 2])
        areas_ok &= len(cells) == 2 and all(abs(c - w) < 1e-6 for c, w in zip(cells, want))
    chk("junction stripes %g-%g um and overlap areas equal Stage 2 targets" % (R["jj_min_um"], R["jj_max_um"]), jr and areas_ok,
        f"{len(L.junctions)} SQUIDs")
    # lengths and ports
    for k, (got, want) in L.lengths.items():
        chk(f"{k} length within 0.2% of target", abs(got - want) / want < 0.002, f"{got:.0f} vs {want:.0f} um")
    xs = sorted(p["x"] for p in L.ports if p["y"] == 0.0)
    chk("launcher pitch >= %g um on the bottom edge" % R["launcher_pitch"], all(b - a >= R["launcher_pitch"] for a, b in zip(xs, xs[1:])), str(xs))
    return out


# ---------------------------------------------------------------- shared readout chain
def solve_resonator(wp_fn, target, X0):
    """Match the target length with the serpentine leg length; if the route is still too long, shrink the
    riser offset X, and as a last resort the coupling run Lc (minimum 100 um). Returns (Lg, X, Lc)."""
    f = lambda Lg, X, Lc: plen(rounded(wp_fn(Lg, X, Lc), R["bend_r"])) - target
    Lmin = 115.
    if f(Lmin, X0, 300.) <= 0:
        return brentq(lambda v: f(v, X0, 300.), Lmin, 2000.), X0, 300.
    if f(Lmin, 20., 300.) <= 0:
        return Lmin, brentq(lambda v: f(Lmin, v, 300.), 20., X0), 300.
    return Lmin, 20., brentq(lambda v: f(Lmin, 20., v), 100., 300.)


def readout_chain(L, tag, res_specs, filter_len_um):
    """res_specs: list of (name, waypoint function, target_um, X0). Filter hairpin + stubs + resonators."""
    cpwg = R["cpw_g"]
    f = lambda xB: plen(rounded([(1000, 8700), (9000, 8700), (9000, 8820), (xB, 8820)], R["bend_r"])) - filter_len_um
    xB = brentq(f, 4000., 8900.)
    sp = L.cpw([(1000, 8700), (9000, 8700), (9000, 8820), (xB, 8820)], "FIL:line", start="open", end="open")
    L.lengths["Purcell filter"] = (plen(sp), filter_len_um)
    pin = L.launcher("IN:launcher", 0, 8700, (1, 0), "readout_in")
    L.cpw([tuple(pin), (980, 8700)], "IN:line", end="open")
    xo = xB - 150
    top = L.launcher("OUT:launcher", xo, H, (0, -1), "readout_out")
    L.cpw([(xB - 20, 8820), (xo, 8820), tuple(top)], "OUT:line", start="open")
    L.strap((790, 8700), (0, 1), 11.); L.strap((xo, 9150), (1, 0), 11.)
    for name, wpf, target, X0 in res_specs:
        Lg, X, Lc = solve_resonator(wpf, target, X0)
        sp = L.cpw(wpf(Lg, X, Lc), f"{name}:res", start="open", end="short")
        L.info_res = getattr(L, "info_res", {}); L.info_res[name] = dict(leg_um=Lg, riser_offset_um=X, filter_run_um=Lc)
        L.lengths[f"{name} resonator"] = (plen(sp), target)
    return xB


def serp(x, y1, Lg, sgn, Lc=300.):
    """Three-leg serpentine tail after the riser point (x, y1): legs toward -sgn*x, then riser to the run."""
    xe = x - sgn * Lg
    return [(xe, y1), (xe, y1 + 100), (x, y1 + 100), (x, y1 + 200), (xe, y1 + 200), (xe, Y_R), (xe + sgn * Lc, Y_R)]


def fmt_targets():
    return json.load(open("stage2_targets.json"))


# ---------------------------------------------------------------- Kähler die
def build_kahler(P, tune=None):
    L = Layout("KAHLER_DIE")
    t = dict(arm_q=R["arm"], arm_cp=R["arm"], head_qc=None, head_qq=None, lq_K0=None, lq_K1=None, len_K0=1., len_K1=1.)
    t.update(tune or {}); L.tune = t
    tr = P["transmons"]["kahler_die"]; s1, s2 = tr["side1_nm"] / 1000, tr["side2_nm"] / 1000
    cc, rs = P["capacitance"]["coupling_fF"], P["capacitance"]["resonators"]
    Jc = B("Jc_uA_per_um2"); cp = P["coupler"]
    Icc = q2.Ic_nA(cp["EJ_sum_GHz"]); d = B("squid_d")
    c1, c2 = (math.sqrt(Icc * (1 - d) / 2 * 1e-3 / Jc), math.sqrt(Icc * (1 + d) / 2 * 1e-3 / Jc))
    y0 = 5000.; xs = {"K0": 4000., "CP": 5000., "K1": 6000.}
    aq, acp = t["arm_q"], t["arm_cp"]; arms = {"K0": aq, "CP": acp, "K1": aq}
    for n, x in xs.items():
        L.cross(f"{n}:island", x, y0, arms[n])
    L.squid(xs["K0"], y0 - aq, s1, s2, "K0"); L.squid(xs["K1"], y0 - aq, s1, s2, "K1"); L.squid(xs["CP"], y0 - acp, c1, c2, "coupler")
    hq = t["head_qc"] or max(8., cc["kahler_qc"] / CPL)
    hd = t["head_qq"] or max(8., cc["kahler_qq"] / CPL)
    # qubit-coupler capacitors (heads face each other across 10 um at x = 4500 and 5500)
    L.wire([(xs["K0"] + aq, y0), (4500 - 13, y0)], "K0:wire", hq); L.wire([(xs["CP"] - acp, y0), (4500 + 13, y0)], "CP:wire", hq)
    L.wire([(xs["CP"] + acp, y0), (5500 - 13, y0)], "CP:wire", hq); L.wire([(xs["K1"] - aq, y0), (5500 + 13, y0)], "K1:wire", hq)
    # direct qubit-qubit capacitor over the coupler
    L.wire([(4300, y0), (4300, 5600), (5000 - 13, 5600)], "K0:wire", hd); L.wire([(5700, y0), (5700, 5600), (5000 + 13, 5600)], "K1:wire", hd)
    # flux (Z) and drive (XY) lines from the bottom edge
    for n, xz, ar in (("K0", 3940., aq), ("CP", 4940., acp), ("K1", 5940., aq)):
        L.launcher(f"{n}Z:launcher", xz, 0, (0, 1), f"{n}_flux")
        wp = [(xz, 600), (xz, y0 - ar - 20)]
        L.cpw(wp, f"{n}Z:line", end="short"); L.strap_line(wp)
    for n, xl, xe in (("K0", 2500., 3550.), ("K1", 7500., 6450.)):
        L.launcher(f"{n}XY:launcher", xl, 0, (0, 1), f"{n}_drive")
        wp = [(xl, 600), (xl, y0), (xe, y0)]
        L.cpw(wp, f"{n}XY:line", end="open"); L.strap_line(wp)
    # readout resonators side-coupled to the top arms, shared Purcell filter
    specs = []
    for n, sgn in (("K0", 1), ("K1", -1)):
        r = rs[n]; lqr = t[f"lq_{n}"] or r["C_qr_fF"] / CPL; x = xs[n] + sgn * 27.
        wpf = (lambda x_, lq, sg: (lambda Lg, X, Lc=300.: [(x_, y0 + aq - lq), (x_, 6200.)] + serp(x_, 6200., Lg, sg, Lc)))(x, lqr, sgn)
        specs.append((n, wpf, r["length_mm"] * 1e3 * t[f"len_{n}"], 0.))
    fl = 2.99792458e8 / (2 * np.mean([rs[k]["f_res_GHz"] for k in rs if k.startswith("K")]) * 1e9 * math.sqrt(B("eps_eff"))) * 1e6
    readout_chain(L, "K", specs, fl)
    marks_and_label(L, "KAHLER DIE (generic rules)")
    L.info = {"coupling_head_um": {"qubit_coupler": hq, "qubit_qubit": hd}, "coupler_junction_sides_nm": [c1 * 1e3, c2 * 1e3]}
    return L


def build_hyper(P, tune=None):
    L = Layout("HYPERKAHLER_DIE")
    t = dict(arm_h=R["arm"], head_h=None, lq_H0=None, lq_H1=None, lq_H2=None, len_H0=1., len_H1=1., len_H2=1.)
    t.update(tune or {}); L.tune = t
    tr = P["transmons"]["hyperkahler_die"]; s1, s2 = tr["side1_nm"] / 1000, tr["side2_nm"] / 1000
    cc, rs = P["capacitance"]["coupling_fF"], P["capacitance"]["resonators"]
    a = t["arm_h"]; pos = {"H0": (3800., 4600.), "H1": (6200., 4600.), "H2": (5000., 5900.)}
    for n, (x, y) in pos.items():
        L.cross(f"{n}:island", x, y, a); L.squid(x, y - a, s1, s2, n)
    hh = t["head_h"] or max(8., cc["hyperkahler_edge"] / CPL)
    (x0, y0), (x1, y1), (x2, y2) = pos["H0"], pos["H1"], pos["H2"]
    zx, xyx = 4940., 5180.
    # H0-H1 wire along y0, interrupted twice by signal airbridges that pass over H2's flux and drive lines
    for xa, xb in ((x0 + a, 4877.), (5003., 5118.), (5242., 5487.)):
        L.add("H0:wire", box(xa, y0 - 4, xb, y0 + 4))
    for xc in (4877., 5003., 5118., 5242.):
        L.add("H0:wire", box(xc - 18, y0 - 18, xc + 18, y0 + 18))
    L.add("H0:wire", box(5487., y0 - hh / 2, 5495., y0 + hh / 2)); L.heads.append(box(5487., y0 - hh / 2, 5495., y0 + hh / 2))
    L.etch.append(box(x0 + a, y0 - 26, 5517., y0 + 26))
    L.crossover((4877., y0), (5003., y0)); L.crossover((5118., y0), (5242., y0))
    L.wire([(x1 - a, y0), (5513., y0)], "H1:wire", hh)
    # H0-H2 and H1-H2 wires
    L.wire([(x0, y0 + a), (x0, y2), (4287., y2)], "H0:wire2", hh); L.wire([(x2 - a, y2), (4313., y2)], "H2:wire", hh)
    L.wire([(x1, y0 + a), (x1, y2), (5713., y2)], "H1:wire2", hh); L.wire([(x2 + a, y2), (5687., y2)], "H2:wire2", hh)
    # flux and drive lines from the bottom edge
    for n, xz in (("H0", 3740.), ("H2", zx), ("H1", 6140.)):
        L.launcher(f"{n}Z:launcher", xz, 0, (0, 1), f"{n}_flux")
        wp = [(xz, 600), (xz, pos[n][1] - a - 20)]
        L.cpw(wp, f"{n}Z:line", end="short"); L.strap_line(wp)
    for n, xl, wp in (("H0", 3000., [(3000, 600), (3000, 4450), (3400, 4450)]),
                      ("H1", 7000., [(7000, 600), (7000, 4450), (6600, 4450)]),
                      ("H2", 5500., [(5500, 600), (5500, 1500), (xyx, 1500), (xyx, 5600)])):
        L.launcher(f"{n}XY:launcher", xl, 0, (0, 1), f"{n}_drive")
        L.cpw(wp, f"{n}XY:line", end="open"); L.strap_line(wp)
    # readout resonators side-coupled to outer/top arms, shared Purcell filter
    specs = []

    def mk(x_, y_, sg, lq):
        def wp(Lg, X, Lc=300.):
            xr = x_ - sg * (a + X)
            return [(x_ - sg * (a - lq), y_ + 27.), (xr, y_ + 27.), (xr, 6500.)] + serp(xr, 6500., Lg, sg, Lc)
        return wp
    for n, sgn in (("H0", 1), ("H1", -1)):
        specs.append((n, mk(pos[n][0], pos[n][1], sgn, t[f"lq_{n}"] or rs[n]["C_qr_fF"] / CPL), rs[n]["length_mm"] * 1e3 * t[f"len_{n}"], 290.))
    lq2 = t["lq_H2"] or rs["H2"]["C_qr_fF"] / CPL
    specs.append(("H2", lambda Lg, X, Lc=300.: [(5027., y2 + a - lq2), (5027., 6800.)] + serp(5027., 6800., Lg, -1, Lc), rs["H2"]["length_mm"] * 1e3 * t["len_H2"], 0.))
    fl = 2.99792458e8 / (2 * np.mean([rs[k]["f_res_GHz"] for k in rs if k.startswith("H")]) * 1e9 * math.sqrt(B("eps_eff"))) * 1e6
    readout_chain(L, "H", specs, fl)
    marks_and_label(L, "HYPERKAHLER DIE (generic rules)")
    L.info = {"coupling_head_um": hh}
    return L


def marks_and_label(L, text):
    for x, y in ((300, 300), (W - 300, 300), (300, H - 300), (W - 300, H - 300)):
        L.etch.append(unary_union([box(x - 100, y - 10, x + 100, y + 10), box(x - 10, y - 100, x + 10, y + 100)]))
    L.text = text


# ---------------------------------------------------------------- output
def to_gds(L, path):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(L.name)

    def put(geom, layer):
        for g in getattr(geom, "geoms", [geom]):
            if g.is_empty or g.geom_type != "Polygon":
                continue
            ext = gdstk.Polygon(list(g.exterior.coords)[:-1], layer=layer)
            if g.interiors:
                cell.add(*gdstk.boolean(ext, [gdstk.Polygon(list(h.coords)[:-1]) for h in g.interiors], "not", layer=layer))
            else:
                cell.add(ext)
    put(L.metal, 1)
    put(unary_union(L.jj1), 2); put(unary_union(L.jj2), 3)
    put(unary_union([st["p1"] for st in L.straps] + [st["p2"] for st in L.straps]), 4)
    put(unary_union([st["span"] for st in L.straps]), 5)
    cell.add(gdstk.rectangle((0, 0), (W, H), layer=6))
    cell.add(*gdstk.text(L.text, 80, (500, 9700), layer=10))
    lib.write_gds(path)


def draw(L, path, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)

    def paint(geom, fc, z=1, alpha=1.0):
        for g in getattr(geom, "geoms", [geom]):
            if g.is_empty or g.geom_type != "Polygon":
                continue
            g = orient(g, 1.0)
            v = list(g.exterior.coords); c = [Path.MOVETO] + [Path.LINETO] * (len(v) - 2) + [Path.CLOSEPOLY]
            for h in g.interiors:
                hv = list(h.coords); v += hv; c += [Path.MOVETO] + [Path.LINETO] * (len(hv) - 2) + [Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(v, c), fc=fc, ec="none", zorder=z, alpha=alpha))
    ax.add_patch(plt.Rectangle((0, 0), W, H, fc="#e8ecef", ec="k", lw=0.8, zorder=0))
    paint(L.metal, "#7f8c99", 1)
    colors = {"island": "#0b3d91", "wire": "#0b3d91", "res": "#1b7f3b", "line": "#c26a00", "launcher": "#c26a00"}
    for n, g in L.net_u.items():
        paint(g, colors.get(kind(n), "#0b3d91"), 2)
    paint(unary_union(L.jj1 + L.jj2).buffer(1.5), "#d00000", 4)
    paint(unary_union([st["span"] for st in L.straps]), "#00a3e0", 3)
    paint(unary_union([st["p1"] for st in L.straps] + [st["p2"] for st in L.straps]), "#ffd400", 3)
    ax.set_xlim(*(xlim or (0, W))); ax.set_ylim(*(ylim or (0, H))); ax.set_aspect("equal")
    ax.set_title(L.name + ("" if xlim is None else " (zoom)") + "  |  gray=ground, blue=islands/wires, green=resonators, orange=lines, red=junctions, cyan/yellow=airbridges", fontsize=7)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def run(L, gds, png, zoom):
    L.build(); L.stitch(); L.build()
    res = drc(L, None)
    for r in res:
        print(f" {r['check']}: {ok(r['pass'])}  ({r['detail']})" if not r["pass"] else f" {r['check']}: {ok(True)}")
    to_gds(L, gds); draw(L, png); draw(L, png.replace(".png", "_zoom.png"), *zoom)
    return res


if __name__ == "__main__":
    P = fmt_targets()
    report = {"rules_assumed": R, "CPL_fF_per_um_assumed": CPL}
    print("Kähler die layout (10 x 10 mm)")
    K = build_kahler(P)
    report["kahler"] = {"drc": run(K, "kahler_die.gds", "kahler_die.png", ((3300, 6700), (4300, 6800))),
                        "ports": K.ports, "junctions": [{k: v for k, v in j.items() if k != "overlap_um2"} for j in K.junctions],
                        "lengths_um": {k: list(v) for k, v in K.lengths.items()}, "info": K.info, "n_airbridges": len(K.straps)}
    print("Hyperkähler die layout (10 x 10 mm)")
    Hh = build_hyper(P)
    report["hyperkahler"] = {"drc": run(Hh, "hyperkahler_die.gds", "hyperkahler_die.png", ((2800, 7300), (4000, 6500))),
                             "ports": Hh.ports, "junctions": [{k: v for k, v in j.items() if k != "overlap_um2"} for j in Hh.junctions],
                             "lengths_um": {k: list(v) for k, v in Hh.lengths.items()}, "info": Hh.info, "n_airbridges": len(Hh.straps)}
    json.dump(report, open("stage3_layout_report.json", "w"), indent=2, default=float)
    print("wrote kahler_die.gds, hyperkahler_die.gds, previews, stage3_layout_report.json")
