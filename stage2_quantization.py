"""Stage 2: circuit quantization. Turns the Stage 1 targets into component values for layout.

  1. Exact transmon (E_J, E_C) from target f01 and anharmonicity, by charge-basis diagonalisation
  2. SQUID flux bias for every operating point, charge dispersion
  3. Junction critical currents and areas (J_c is foundry-dependent: swept)
  4. Target Maxwell capacitance matrices for both dies, readout resonator length and coupling capacitor
  5. Tunable-coupler design: coupler E_J, zero-coupling and CZ frequencies
  6. Residual ZZ at idle, computed from the full coupled Hamiltonian

Generic baseline only. Writes stage2_targets.json. Run: python stage2_quantization.py
"""
import itertools
import json
import numpy as np
from scipy.optimize import brentq, fsolve
import chipset as cs
from chipset import ALPHA, B, SPEC, TWO_PI, ok

H_PLANCK, E_CHG, PHI0 = 6.62607015e-34, 1.602176634e-19, 2.067833848e-15
C_LIGHT = 2.99792458e8
D_SQ = B("squid_d")


# ---------------------------------------------------------------- 1. exact transmon
def levels(EJ, EC, ng=0.0, ncut=30, n=4):
    N = np.arange(-ncut, ncut + 1)
    H = np.diag(4 * EC * (N - ng) ** 2) - EJ / 2 * (np.eye(2 * ncut + 1, k=1) + np.eye(2 * ncut + 1, k=-1))
    E = np.linalg.eigvalsh(H)
    return E[:n] - E[0]


def f01_alpha(EJ, EC):
    E = levels(EJ, EC)
    return E[1], (E[2] - E[1]) - E[1]


def solve_EJ_EC(f01, alpha):
    """GHz. Returns (EJ, EC) giving exactly f01 and alpha (alpha<0)."""
    x0 = [(f01 + abs(alpha)) ** 2 / (8 * abs(alpha)), abs(alpha)]
    sol = fsolve(lambda x: np.array(f01_alpha(*x)) - [f01, alpha], x0, xtol=1e-12)
    return float(sol[0]), float(sol[1])


def EJ_at(f01, EC):
    return brentq(lambda ej: f01_alpha(ej, EC)[0] - f01, 1.0, 200.0, xtol=1e-12)


def charge_dispersion_kHz(EJ, EC):
    return abs(levels(EJ, EC, ng=0.0)[1] - levels(EJ, EC, ng=0.5)[1]) * 1e6


def Ic_nA(EJ_GHz):
    return 2 * np.pi * H_PLANCK * EJ_GHz * 1e9 / PHI0 * 1e9


def C_fF(EC_GHz):
    return E_CHG ** 2 / (2 * H_PLANCK * EC_GHz * 1e9) * 1e15


def squid_EJ(EJsum, flux, d=D_SQ):
    x = np.pi * flux
    return EJsum * (np.cos(x) ** 2 + d ** 2 * np.sin(x) ** 2) ** 0.5


def flux_for_EJ(EJ, EJsum, d=D_SQ):
    return brentq(lambda p: squid_EJ(EJsum, p, d) - EJ, 0, 0.5)


# ---------------------------------------------------------------- dressed-state ZZ
def dressed_zz(f, al, G, d=3):
    """ZZ (MHz) for every qubit pair from the full coupled Hamiltonian; modes in GHz."""
    n = len(f); a = np.diag(np.sqrt(np.arange(1, d)), 1); I = np.eye(d)

    def op(k, M):
        out = np.eye(1)
        for j in range(n):
            out = np.kron(out, M if j == k else I)
        return out
    num = lambda k: op(k, a.T @ a)
    H = sum(f[k] * num(k) + al[k] / 2 * (num(k) @ num(k) - num(k)) for k in range(n))
    for (i, j), g in G.items():
        H = H + g * (op(i, a.T) @ op(j, a) + op(i, a) @ op(j, a.T))
    E, V = np.linalg.eigh(H)

    def en(occ):
        idx = sum(o * d ** (n - 1 - k) for k, o in enumerate(occ))
        return E[np.argmax(abs(V[idx, :]) ** 2)]
    out = {}
    for i, j in itertools.combinations(range(n), 2):
        def occ(x, y):
            o = [0] * n; o[i] = x; o[j] = y; return o
        out[(i, j)] = (en(occ(1, 1)) - en(occ(1, 0)) - en(occ(0, 1)) + en(occ(0, 0))) * 1e3
    return out


# ---------------------------------------------------------------- coupler effective coupling
def g_eff(w1, w2, wc, gd, g1, g2):
    """Second-order qubit-qubit coupling through a coupler (MHz in, MHz out; w in GHz)."""
    return gd + 0.5 * g1 * g2 * (1 / ((w1 - wc) * 1e3) + 1 / ((w2 - wc) * 1e3))


def g_eff_exact(w, wc, gd, g1, g2):
    """Exact single-excitation check with both qubits at w (MHz): half the sym/antisym splitting."""
    g = np.sqrt(g1 * g2) * 1e-3; gd_ = gd * 1e-3
    Ea = w - gd_
    Es = np.linalg.eigvalsh(np.array([[w + gd_, np.sqrt(2) * g], [np.sqrt(2) * g, wc]]))
    Es = Es[np.argmin(abs(Es - (w + gd_)))]
    return (Es - Ea) / 2 * 1e3


# ---------------------------------------------------------------- main parts
def part_transmons():
    print("Exact transmon quantization (charge-basis diagonalisation, target f01 and α exact)")
    out = {}
    ec_ref = None
    for die in ("kahler_die", "hyperkahler_die"):
        fm = SPEC[die]["qubits"][0]["f_max"]
        EJs, EC = solve_EJ_EC(fm, ALPHA * 1e-3)
        ec_ref = EC
        EJ1, EJ2 = EJs * (1 - D_SQ) / 2, EJs * (1 + D_SQ) / 2
        Ic1, Ic2 = Ic_nA(EJ1), Ic_nA(EJ2)
        Jc = B("Jc_uA_per_um2")
        area = lambda ic: ic * 1e-3 / Jc                                # um^2
        res = {"f_max_GHz": fm, "EJ_sum_GHz": EJs, "EC_MHz": EC * 1e3, "EJ_over_EC": EJs / EC,
               "C_sigma_fF": C_fF(EC), "EJ1_GHz": EJ1, "EJ2_GHz": EJ2, "Ic1_nA": Ic1, "Ic2_nA": Ic2,
               "area1_um2": area(Ic1), "area2_um2": area(Ic2),
               "side1_nm": 1e3 * np.sqrt(area(Ic1)), "side2_nm": 1e3 * np.sqrt(area(Ic2)),
               "sides_over_Jc_sweep_nm": {f"{j}": [1e3 * np.sqrt(Ic1 * 1e-3 / j), 1e3 * np.sqrt(Ic2 * 1e-3 / j)]
                                          for j in PARAMS_JC}}
        out[die] = res
        print(f" {die}: f_max {fm} GHz -> EJΣ={EJs:.2f} GHz, EC={EC * 1e3:.1f} MHz (EJ/EC {EJs / EC:.0f}), C_Σ={C_fF(EC):.0f} fF | "
              f"SQUID Ic {Ic1:.1f} / {Ic2:.1f} nA -> junctions {res['side1_nm']:.0f} / {res['side2_nm']:.0f} nm square at Jc={Jc} µA/µm²")
    d0 = out["kahler_die"]
    print(f" exact EC differs from the Stage 1 value {cs.EC} MHz by {abs(d0['EC_MHz'] - cs.EC):.1f} MHz "
          f"(Stage 1 used EC ≈ -α)")
    print(f" EJ/EC > 50 and junction sides 100-400 nm at baseline Jc and all swept Jc: "
          f"{ok(all(v['EJ_over_EC'] > 50 and 100 < min(sum(v['sides_over_Jc_sweep_nm'].values(), [])) and max(sum(v['sides_over_Jc_sweep_nm'].values(), [])) < 400 for v in out.values()))}")
    return out


PARAMS_JC = (B("Jc_uA_per_um2"), *cs.PARAMS["Jc_uA_per_um2"][1])


def part_flux_and_dispersion(tr):
    print("Flux bias and charge dispersion at every operating point (exact)")
    pts = {}
    for die in ("kahler_die", "hyperkahler_die"):
        for q in SPEC[die]["qubits"]:
            pts[q["id"] + "_park"] = (q["f_park"], die)
    pts["K1_CZ"] = (SPEC["kahler_die"]["qubits"][0]["f_park"] + ALPHA * 1e-3, "kahler_die")
    pts["H_operate"] = (SPEC["hyperkahler_die"]["operating_freq"], "hyperkahler_die")
    out, worst_disp, worst_alpha = {}, 0.0, 0.0
    for name, (f, die) in pts.items():
        t = tr[die]; EC = t["EC_MHz"] * 1e-3
        EJ = EJ_at(f, EC)
        flux = flux_for_EJ(EJ, t["EJ_sum_GHz"])
        al = f01_alpha(EJ, EC)[1] * 1e3
        disp = charge_dispersion_kHz(EJ, EC)
        out[name] = {"f01_GHz": f, "flux_Phi0": float(flux), "alpha_MHz": float(al), "charge_dispersion_kHz": float(disp)}
        worst_disp = max(worst_disp, disp); worst_alpha = max(worst_alpha, abs(al - ALPHA))
        print(f" {name:<10} {f:.3f} GHz: flux {flux:.3f} Φ0, α={al:.1f} MHz, charge dispersion {disp:.1e} kHz")
    lw = 1e-3 / (2 * np.pi * B("T2_us") * 1e-6)                          # kHz: 1/(2 pi T2)
    print(f" charge dispersion below the dephasing linewidth 1/(2πT2) = {lw:.1f} kHz everywhere (worst {worst_disp:.1f} kHz, "
          f"marginal at the lowest parking point): {ok(worst_disp < lw)}")
    print(f" anharmonicity stays within 5% of {ALPHA} MHz when tuned (worst {worst_alpha:.1f} MHz, calibrated out): "
          f"{ok(worst_alpha < 0.05 * abs(ALPHA))}")
    import stage1_budget as s1
    dflux = max(abs(out[k]["flux_Phi0"] - s1.flux_point(v[0], SPEC[v[1]]["qubits"][0]["f_max"], D_SQ))
                for k, v in pts.items() if k != "K1_CZ" and k != "H_operate")
    print(f" Stage 1 approximate flux points agree with exact within 0.005 Φ0 (worst {dflux:.4f}): {ok(dflux < 0.005)}")
    return out


def part_capacitance(tr):
    print("Target capacitances (Maxwell matrix, fF) and readout resonators")
    out = {}
    # hyperkahler triangle at the operating frequency
    Ch = tr["hyperkahler_die"]["C_sigma_fF"]; g = B("g_tri_MHz") * 1e-3; w = SPEC["hyperkahler_die"]["operating_freq"]
    Cc = 2 * g * Ch / w
    M = np.diag([Ch] * 3) - Cc * (np.ones((3, 3)) - np.eye(3))
    out["hyperkahler_maxwell_fF"] = M.round(3).tolist()
    print(f" hyperkähler: C_Σ={Ch:.1f} fF each, coupling capacitor {Cc:.3f} fF per edge for g={B('g_tri_MHz')} MHz at {w} GHz")
    # kahler: qubit-qubit direct, qubit-coupler
    Ck = tr["kahler_die"]["C_sigma_fF"]; Ccp = B("coupler_C_fF")
    wk = [q["f_park"] for q in SPEC["kahler_die"]["qubits"]]; wcp = B("coupler_fmax_GHz")
    gqq, gqc = B("g_direct_MHz") * 1e-3, B("g_qc_MHz") * 1e-3
    Cqq = 2 * gqq * Ck / np.sqrt(wk[0] * wk[1])
    Cqc = 2 * gqc * np.sqrt(Ck * Ccp) / np.sqrt(np.mean(wk) * wcp)
    Mk = np.array([[Ck, -Cqq, -Cqc], [-Cqq, Ck, -Cqc], [-Cqc, -Cqc, Ccp]])
    out["kahler_maxwell_fF"] = Mk.round(3).tolist()
    print(f" Kähler: C_Σ={Ck:.1f} fF, coupler {Ccp:.0f} fF; direct qubit-qubit {Cqq:.3f} fF, qubit-coupler {Cqc:.2f} fF")
    # readout resonators
    Cl = B("cpw_C_fF_per_um"); eps = B("eps_eff"); gqr = B("g_qr_MHz") * 1e-3
    res = {}
    for die, key in (("kahler_die", "C_sigma_fF"), ("hyperkahler_die", "C_sigma_fF")):
        for q in SPEC[die]["qubits"]:
            fr = q["readout_res"]; L = C_LIGHT / (4 * fr * 1e9 * np.sqrt(eps)) * 1e3     # mm
            Cr = Cl * L * 1e3 / 2                                                          # fF
            Cqr = 2 * gqr * np.sqrt(tr[die][key] * Cr) / np.sqrt(q["f_park"] * fr)
            res[q["id"]] = {"f_res_GHz": fr, "length_mm": float(L), "C_res_fF": float(Cr), "C_qr_fF": float(Cqr)}
            print(f" {q['id']}: λ/4 resonator {fr:.2f} GHz, length {L:.2f} mm, coupling capacitor {Cqr:.1f} fF")
    out["resonators"] = res
    cq = [v["C_qr_fF"] for v in res.values()]
    print(f" coupling capacitors in a buildable range (1-12 fF): {ok(min(cq) > 1 and max(cq) < 12)}")
    out["coupling_fF"] = {"hyperkahler_edge": float(Cc), "kahler_qq": float(Cqq), "kahler_qc": float(Cqc)}
    return out


def part_coupler(tr):
    print("Tunable coupler (Kähler die)")
    wk = [q["f_park"] for q in SPEC["kahler_die"]["qubits"]]
    gd, g1 = B("g_direct_MHz"), B("g_qc_MHz")
    fmax_c, Cc = B("coupler_fmax_GHz"), B("coupler_C_fF")
    ECc = 19.3664 / Cc                                                    # GHz, e^2/2C
    EJc, ECc = solve_EJ_EC(fmax_c, -ECc)
    fmin_c = f01_alpha(EJc * np.sqrt(D_SQ ** 2), ECc)[0]                  # at half flux quantum
    print(f" coupler: C={Cc:.0f} fF, EC={ECc * 1e3:.0f} MHz, EJΣ={EJc:.1f} GHz, f_max {fmax_c} GHz, f_min {fmin_c:.2f} GHz")
    gz = lambda wc: g_eff(wk[0], wk[1], wc, gd, g1, g1)
    wc_zero = brentq(gz, 5.0, 6.9)
    wk_cz = [wk[0], wk[0] + ALPHA * 1e-3]
    wc_cz = brentq(lambda wc: g_eff(wk_cz[0], wk_cz[1], wc, gd, g1, g1) + B("g_cz_eff_MHz"), 4.9, 6.9)
    ex = g_eff_exact(4.7, wc_cz, gd, g1, g1); ap = g_eff(4.7, 4.7, wc_cz, gd, g1, g1)
    print(f" zero coupling at coupler = {wc_zero:.2f} GHz; CZ point (g_eff = -{B('g_cz_eff_MHz')} MHz) at {wc_cz:.2f} GHz")
    print(f" second-order g_eff {ap:.1f} MHz vs exact diagonalisation {ex:.1f} MHz at the CZ point")
    print(f" both coupler frequencies inside [f_min, f_max-10 MHz], second-order within 15% of exact: "
          f"{ok(fmin_c < wc_cz < wc_zero < fmax_c - 0.01 and abs(ap - ex) < 0.15 * abs(ex))}")
    # residual ZZ at the zero-coupling point and best point of a scan
    al = [ALPHA * 1e-3, ALPHA * 1e-3, -ECc]
    G = lambda: {(0, 1): gd * 1e-3, (0, 2): g1 * 1e-3, (1, 2): g1 * 1e-3}
    scan = np.linspace(wc_cz + 0.1, fmax_c - 0.02, 60)
    zz = [dressed_zz([wk[0], wk[1], w], al, G())[(0, 1)] for w in scan]
    zz0 = dressed_zz([wk[0], wk[1], wc_zero], al, G())[(0, 1)]
    best = scan[int(np.argmin(abs(np.array(zz))))]
    print(f" idle ZZ at zero coupling {zz0:+.3f} MHz; smallest |ZZ| in the coupler scan {min(abs(np.array(zz))):.3f} MHz at {best:.2f} GHz")
    print(f" idle ZZ < {B('zz_tol_MHz')} MHz achievable with the coupler: {ok(min(abs(np.array(zz))) < B('zz_tol_MHz'))}")
    return {"EJ_sum_GHz": EJc, "EC_MHz": ECc * 1e3, "f_min_GHz": float(fmin_c), "f_zero_coupling_GHz": float(wc_zero),
            "f_cz_GHz": float(wc_cz), "zz_at_zero_coupling_MHz": float(zz0), "min_abs_zz_MHz": float(min(abs(np.array(zz)))),
            "f_min_zz_GHz": float(best)}


def part_triangle_zz():
    print("Residual ZZ at idle, hyperkähler die (full coupled Hamiltonian, 4 levels)")
    f = [q["f_park"] for q in SPEC["hyperkahler_die"]["qubits"]]
    g = B("g_tri_MHz") * 1e-3
    z = dressed_zz(f, [ALPHA * 1e-3] * 3, {(0, 1): g, (0, 2): g, (1, 2): g}, d=4)
    for k, v in z.items():
        print(f" H{k[0]}-H{k[1]}: {v:+.2f} MHz")
    print(f" all |ZZ| < {B('zz_tol_MHz')} MHz: {ok(max(abs(v) for v in z.values()) < B('zz_tol_MHz'))}")
    return {f"H{a}-H{b}": float(v) for (a, b), v in z.items()}


if __name__ == "__main__":
    tr = part_transmons()
    out = {"transmons": tr, "operating_points": part_flux_and_dispersion(tr), "capacitance": part_capacitance(tr),
           "coupler": part_coupler(tr), "hyperkahler_idle_zz_MHz": part_triangle_zz()}
    json.dump(out, open("stage2_targets.json", "w"), indent=2)
    print("wrote stage2_targets.json")
