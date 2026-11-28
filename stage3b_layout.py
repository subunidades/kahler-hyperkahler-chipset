"""Stage 3b: compact re-layout of both dies, to fit the Stage 2 capacitance budget.

Stage 4 showed that the Stage 3 floorplan carries 1.5 to 4.2 mm of conductor per qubit, which costs about
0.1 fF per um, against a budget of roughly 880 um. This version packs the islands about 700 um apart,
couples them through short arm tips, uses narrower arms and a wider ground moat, and exposes a spare arm
(and a pad on the coupler) as the knobs that Stage 4 tunes to hit the capacitance targets.

Same design rules and same checks as Stage 3 (generic rules, not a foundry's).
Run: python stage3b_layout.py   (needs stage2_targets.json); writes kahler_die_v2.gds, hyperkahler_die_v2.gds.
"""
import json
import math
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union
import stage3_layout as s3
from stage3_layout import B, CPL, H, R, W, Y_R, Layout, ok, q2, rect, rounded, serp, unit

# Final values from the Stage 4 tuning loop (see stage4_extract.py); builders start from these.
TUNED_K = {'head_qc': 35.63, 'gap_qq': 40.85, 'lq_K0': 117.63, 'lq_K1': 130.47, 'aqL': 13.33, 'pad_cp': 88.3, 'len_K0': 0.99177, 'len_K1': 0.99163}
TUNED_H = {'gap_h02': 118.94, 'gap_h12': 57.46, 'gap01': 21.46, 'lq_H0': 123.63, 'lq_H1': 116.6, 'lq_H2': 93.6, 'aL0': 190.0, 'aL1': 246.2, 'aH2U': 443.7, 'len_H0': 0.99162, 'len_H1': 0.99146, 'len_H2': 0.99211}

DIRS = dict(L=(-1, 0), R=(1, 0), U=(0, 1), D=(0, -1))


class Layout2(Layout):
    def __init__(s, name):
        super().__init__(name)
        s.head_keepout = 70.

    def island(s, net, cx, cy, arms, w=12., pad=0., clear=40.):
        """Cross with independent arm lengths (L, R, U, D from the centre), optional square pad of half-size `pad`."""
        parts = [box(cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2)]
        for k, (dx, dy) in DIRS.items():
            a = arms.get(k, 0)
            if a > 0:
                x1, y1 = cx + dx * a, cy + dy * a
                parts.append(box(min(cx, x1) - (w / 2 if dx == 0 else 0), min(cy, y1) - (w / 2 if dy == 0 else 0),
                                 max(cx, x1) + (w / 2 if dx == 0 else 0), max(cy, y1) + (w / 2 if dy == 0 else 0)))
        if pad:
            parts.append(box(cx - pad, cy - pad, cx + pad, cy + pad))
        p = unary_union(parts); s.add(net, p)
        e = p.buffer(clear, join_style=2)
        if arms.get("D", 0) > 0 and clear > 30:                      # keep ground close below the SQUID tip
            ytip = cy - max(arms["D"], pad)
            e = e.difference(box(cx - 90, ytip - clear - 1, cx + 90, ytip - 30))
        s.etch.append(e)

    def wire2(s, pts, net, head=8., w=8., clear=22., r=30.):
        sp = rounded(pts, r); ls = LineString(sp); u = unit(sp[-1] - sp[-2])
        hd = rect(sp[-1] + u * 4, u, 8., head)
        s.add(net, ls.buffer(w / 2, cap_style=2, join_style=1)); s.add(net, hd); s.heads.append(hd)
        s.etch.append(unary_union([ls.buffer(w / 2 + clear, cap_style=2, join_style=1), hd.buffer(clear, join_style=2)]))


def _junctions(P):
    tr = P["transmons"]["kahler_die"]
    Jc, d = B("Jc_uA_per_um2"), B("squid_d")
    Icc = q2.Ic_nA(P["coupler"]["EJ_sum_GHz"])
    return tr, (math.sqrt(Icc * (1 - d) / 2 * 1e-3 / Jc), math.sqrt(Icc * (1 + d) / 2 * 1e-3 / Jc))


# ---------------------------------------------------------------- Kähler die
def build_kahler(P, tune=None):
    L = Layout2("KAHLER_DIE_V2")
    t = dict(D=480., aqR=60., aqU=170., aqD=30., aqL=30., pad_cp=70., head_qc=None, head_qq=None, gap_qq=10., lq_K0=None, lq_K1=None,
             len_K0=1., len_K1=1., clear=40.)
    t.update(TUNED_K); t.update(tune or {}); L.tune = t
    tr, (c1, c2) = _junctions(P)
    s1, s2 = tr["side1_nm"] / 1000, tr["side2_nm"] / 1000
    cc, rs = P["capacitance"]["coupling_fF"], P["capacitance"]["resonators"]
    y0, xc, D, ps, cl = 5000., 5000., t["D"], t["pad_cp"], t["clear"]
    xK0, xK1 = xc - D, xc + D
    aR, aU, aD, aL = t["aqR"], t["aqU"], t["aqD"], t["aqL"]
    aC, aCD, aCU = ps + 30, ps + 40, ps + 10
    L.island("K0:island", xK0, y0, dict(R=aR, U=aU, D=aD, L=aL), clear=cl)
    L.island("K1:island", xK1, y0, dict(L=aR, U=aU, D=aD, R=aL), clear=cl)
    L.island("CP:island", xc, y0, dict(L=aC, R=aC, D=aCD, U=aCU), pad=ps, clear=cl)
    L.squid(xK0, y0 - aD, s1, s2, "K0"); L.squid(xK1, y0 - aD, s1, s2, "K1"); L.squid(xc, y0 - aCD, c1, c2, "coupler")
    hq = t["head_qc"] or max(8., cc["kahler_qc"] / CPL)
    hd = t["head_qq"] or max(8., cc["kahler_qq"] / CPL)
    m1 = (xK0 + aR + xc - aC) / 2
    L.wire2([(xK0 + aR, y0), (m1 - 13, y0)], "K0:wire", hq); L.wire2([(xc - aC, y0), (m1 + 13, y0)], "CP:wire", hq)
    L.wire2([(xK1 - aR, y0), (2 * xc - m1 + 13, y0)], "K1:wire", hq); L.wire2([(xc + aC, y0), (2 * xc - m1 - 13, y0)], "CP:wire", hq)
    yb = y0 + max(ps, aCU) + cl + 40
    xt = xK0 + aR + 20
    he = t["gap_qq"] / 2 + 8
    L.wire2([(xt, y0), (xt, yb), (xc - he, yb)], "K0:wire", hd); L.wire2([(2 * xc - xt, y0), (2 * xc - xt, yb), (xc + he, yb)], "K1:wire", hd)
    for n, xz, tip in (("K0", xK0 - 80., y0 - aD), ("CP", xc - 60., y0 - aCD), ("K1", xK1 + 80., y0 - aD)):
        L.launcher(f"{n}Z:launcher", xz, 0, (0, 1), f"{n}_flux")
        wp = [(xz, 600), (xz, tip - 50)]
        L.cpw(wp, f"{n}Z:line", end="short"); L.strap_line(wp)
    L.etch.append(box(xc - he - 8, yb - 26, xc + he + 8, yb + 26))      # merge the moats of the two heads
    for n, xl, xe in (("K0", 3200., xK0 - aL - 150), ("K1", 6800., xK1 + aL + 150)):
        L.launcher(f"{n}XY:launcher", xl, 0, (0, 1), f"{n}_drive")
        wp = [(xl, 600), (xl, y0), (xe, y0)]
        L.cpw(wp, f"{n}XY:line", end="open"); L.strap_line(wp)
    specs = []
    for n, xq, sgn in (("K0", xK0, 1), ("K1", xK1, -1)):
        r = rs[n]; lqr = t[f"lq_{n}"] or r["C_qr_fF"] / CPL; x = xq - sgn * 21.
        wpf = (lambda x_, lq, sg: (lambda Lg, X, Lc=300.: [(x_, y0 + aU - lq), (x_, 6200.)] + serp(x_, 6200., Lg, sg, Lc)))(x, lqr, sgn)
        specs.append((n, wpf, r["length_mm"] * 1e3 * t[f"len_{n}"], 0.))
    fl = 2.99792458e8 / (2 * np.mean([rs[k]["f_res_GHz"] for k in rs if k.startswith("K")]) * 1e9 * math.sqrt(B("eps_eff"))) * 1e6
    s3.readout_chain(L, "K", specs, fl)
    s3.marks_and_label(L, "KAHLER DIE V2 (generic rules)")
    L.win = {"K0": (xK0 + 100, 5150, 1152, 960), "CP": (xc, 5150, 1152, 864), "K1": (xK1 - 100, 5150, 1152, 960)}
    L.info = {"coupling_head_um": {"qubit_coupler": hq, "qubit_qubit": hd}}
    return L


# ---------------------------------------------------------------- hyperkähler die
def build_hyper(P, tune=None):
    L = Layout2("HYPERKAHLER_DIE_V2")
    t = dict(D=320., dy=270., aR=50., aU=30., aL0=190., aL1=190., aD=35., aH2=50., aH2U=330., aH2D=150., pad_h2=50., head_h=6., head_h01=6., gap01=10., gap_h02=10., gap_h12=10.,
             lq_H0=None, lq_H1=None, lq_H2=None, len_H0=1., len_H1=1., len_H2=1., clear=40.)
    t.update(TUNED_H); t.update(tune or {}); L.tune = t
    tr = P["transmons"]["hyperkahler_die"]; s1, s2 = tr["side1_nm"] / 1000, tr["side2_nm"] / 1000
    cc, rs = P["capacitance"]["coupling_fF"], P["capacitance"]["resonators"]
    cl = t["clear"]; xc, y0 = 5000., 4600.; y2 = y0 + t["dy"]
    xH0, xH1 = xc - t["D"], xc + t["D"]
    aR, aU, aD = t["aR"], t["aU"], t["aD"]; aL0, aL1 = t["aL0"], t["aL1"]; ph = t["pad_h2"]
    aH2, aH2U, aH2D = max(t["aH2"], ph + 10), t["aH2U"], max(t["aH2D"], ph + 10)
    L.island("H0:island", xH0, y0, dict(R=aR, U=aU, L=aL0, D=aD), clear=cl)
    L.island("H1:island", xH1, y0, dict(L=aR, U=aU, R=aL1, D=aD), clear=cl)
    L.island("H2:island", xc, y2, dict(L=aH2, R=aH2, U=aH2U, D=aH2D), pad=ph, clear=cl)
    for n, (x, y, d_) in (("H0", (xH0, y0, aD)), ("H1", (xH1, y0, aD)), ("H2", (xc, y2, aH2D))):
        L.squid(x, y - d_, s1, s2, n)
    hh = t["head_h"] or max(8., cc["hyperkahler_edge"] / CPL)
    # H0-H1 link: two signal airbridges let H2's flux and drive lines pass under it
    hh01 = t["head_h01"] or hh
    zx, xyx = xc - 60., xc + 160.
    h1e = zx + 103 + t["gap01"]                                  # east edge of the H1 head
    for net, xa, xb in (("H0:wire", xH0 + aR, zx - 63), ("H0:wire", zx + 63, zx + 87), ("H1:wire", xyx + 63, xH1 - aR), ("H1:wire", xyx - 63, h1e)):
        L.add(net, box(min(xa, xb), y0 - 4, max(xa, xb), y0 + 4))
    for xcn, net in ((zx - 63, "H0:wire"), (zx + 63, "H0:wire"), (xyx + 63, "H1:wire"), (xyx - 63, "H1:wire")):
        L.add(net, box(xcn - 18, y0 - 18, xcn + 18, y0 + 18))
    for net, xa, xb in (("H0:wire", zx + 87, zx + 95), ("H1:wire", h1e - 8, h1e)):
        hbox = box(xa, y0 - hh01 / 2, xb, y0 + hh01 / 2); L.add(net, hbox); L.heads.append(hbox)
    L.etch.append(box(xH0 + aR, y0 - 26, xH1 - aR, y0 + 26))
    L.crossover((zx - 63, y0), (zx + 63, y0)); L.crossover((xyx + 63, y0), (xyx - 63, y0))
    # H0-H2 and H1-H2 links
    hg0, hg1 = t["gap_h02"] / 2 + 8, t["gap_h12"] / 2 + 8
    L.wire2([(xH0, y0 + aU), (xH0, y2), (4800. - hg0, y2)], "H0:wire2", hh); L.wire2([(xc - aH2, y2), (4800. + hg0, y2)], "H2:wire", hh)
    L.wire2([(xH1, y0 + aU), (xH1, y2), (5200. + hg1, y2)], "H1:wire2", hh); L.wire2([(xc + aH2, y2), (5200. - hg1, y2)], "H2:wire2", hh)
    # flux and drive lines from the bottom edge (jogs keep the launchers 500 um apart)
    lines = [("H0Z", [(4100, 600), (4100, 1000), (xH0 - 60, 1000), (xH0 - 60, y0 - aD - 50)], "short", 4100.),
             ("H2Z", [(zx, 600), (zx, y2 - aH2D - 50)], "short", zx),
             ("H2XY", [(5600, 600), (5600, 1500), (xyx, 1500), (xyx, y2 - 120)], "open", 5600.),
             ("H1Z", [(6200, 600), (6200, 2000), (xH1 + 60, 2000), (xH1 + 60, y0 - aD - 50)], "short", 6200.),
             ("H0XY", [(3200, 600), (3200, y0 - 150), (xH0 - aL0 - 120, y0 - 150)], "open", 3200.),
             ("H1XY", [(7000, 600), (7000, y0 - 150), (xH1 + aL1 + 120, y0 - 150)], "open", 7000.)]
    for name, wp, end, xl in lines:
        L.launcher(f"{name}:launcher", xl, 0, (0, 1), f"{name}_port")
        L.cpw(wp, f"{name}:line", end=end); L.strap_line(wp)
    # readout resonators: H0 and H1 along the upper side of their outer arms, H2 along its top arm
    specs = []

    def mk(x_, sg, lq, aLn):
        def wp(Lg, X, Lc=300.):
            xr = x_ - sg * (aLn + X)
            return [(x_ - sg * (aLn - lq), y0 + 21.), (xr, y0 + 21.), (xr, 6500.)] + serp(xr, 6500., Lg, sg, Lc)
        return wp
    for n, x_, sg, aLn in (("H0", xH0, 1, aL0), ("H1", xH1, -1, aL1)):
        specs.append((n, mk(x_, sg, t[f"lq_{n}"] or rs[n]["C_qr_fF"] / CPL, aLn), rs[n]["length_mm"] * 1e3 * t[f"len_{n}"], 290.))
    lq2 = t["lq_H2"] or rs["H2"]["C_qr_fF"] / CPL
    specs.append(("H2", lambda Lg, X, Lc=300.: [(xc + 21., y2 + aH2U - lq2), (xc + 21., 6800.)] + serp(xc + 21., 6800., Lg, -1, Lc),
                  rs["H2"]["length_mm"] * 1e3 * t["len_H2"], 0.))
    fl = 2.99792458e8 / (2 * np.mean([rs[k]["f_res_GHz"] for k in rs if k.startswith("H")]) * 1e9 * math.sqrt(B("eps_eff"))) * 1e6
    s3.readout_chain(L, "H", specs, fl)
    s3.marks_and_label(L, "HYPERKAHLER DIE V2 (generic rules)")
    L.win = {"H0": (xH0 - 100, 4750, 1152, 960), "H1": (xH1 + 100, 4750, 1152, 960), "H2": (xc, 5000, 864, 960), "HE": (xc + 40, y0, 384, 192)}
    L.info = {"coupling_head_um": hh}
    return L


if __name__ == "__main__":
    P = s3.fmt_targets(); report = {}
    for tag, build, zoom in (("kahler", build_kahler, ((4000, 6000), (4300, 5600))), ("hyperkahler", build_hyper, ((4000, 6000), (4300, 5400)))):
        print(f"{tag} die v2 layout (10 x 10 mm)")
        L = build(P)
        res = s3.run(L, f"{tag}_die_v2.gds", f"{tag}_die_v2.png", zoom)
        report[tag] = {"drc": res, "ports": L.ports, "lengths_um": {k: list(v) for k, v in L.lengths.items()}, "n_airbridges": len(L.straps)}
    json.dump(report, open("stage3b_layout_report.json", "w"), indent=2, default=float)
    print("wrote kahler_die_v2.gds, hyperkahler_die_v2.gds, previews, stage3b_layout_report.json")
