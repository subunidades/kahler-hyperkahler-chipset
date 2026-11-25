"""Stage 1: circuit-level completion and per-gate error budget for both dies.

  1. SQUID flux response and 1/f flux noise at each operating point (Ithier-style model)
  2. DRAG single-qubit gate, 3-level transmon, optimised pulse, with and without DRAG
  3. Dispersive readout, Purcell limit with and without a filter, readout error
  4. Per-gate error budget; native-gate errors in the Qiskit Target are updated from it

Inputs are the generic baseline in chipset.PARAMS (not foundry data, not measured).
Run: python stage1_budget.py   (writes stage1_budget.json)
"""
import contextlib
import io
import json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, minimize
from scipy.special import erfc
from qiskit import transpile
import chipset as cs
from chipset import ALPHA, B, EC, SPEC, TWO_PI, coherence_error, ok, transmon
import hyperkahler_unit as hk

EC_G = EC * 1e-3                                                     # GHz


# ---------------------------------------------------------------- 1. flux response and noise
def squid_freq(flux, fmax, d):
    """f01(Phi/Phi0) of an asymmetric-SQUID transmon, GHz."""
    x = np.pi * flux
    return (fmax + EC_G) * (np.cos(x) ** 2 + d ** 2 * np.sin(x) ** 2) ** 0.25 - EC_G


def flux_point(f, fmax, d):
    fmin = squid_freq(0.5, fmax, d)
    if not (fmin < f <= fmax):
        raise ValueError(f"{f} GHz outside tunable range [{fmin:.2f}, {fmax:.2f}]")
    return brentq(lambda p: squid_freq(p, fmax, d) - f, 0, 0.5)


def flux_noise(f, fmax, d, t_gate_ns=None):
    """Slope, 1/f echo dephasing and quasi-static frequency jitter at an operating point."""
    p = flux_point(f, fmax, d) if f < fmax - 1e-9 else 0.0
    h = 1e-6
    slope = abs(squid_freq(p + h, fmax, d) - squid_freq(p - h, fmax, d)) / (2 * h)
    A = (B("A_phi_uPhi0") * 1e-6) ** 2                               # Phi0^2 at 1 Hz
    s_rad = slope * TWO_PI * 1e9                                     # rad/s per Phi0
    g_echo = s_rad * np.sqrt(A * np.log(2))                          # exp(-(t*g)^2)
    out = {"f_GHz": f, "flux_Phi0": float(p), "slope_GHz_per_Phi0": float(slope),
           "Tphi_echo_us": float(1e6 / g_echo) if g_echo > 0 else float("inf"),
           "sigma_f_quasistatic_MHz": float(slope * np.sqrt(A * np.log((1 / 200e-9) * 3600)) * 1e3)}
    if t_gate_ns:                                                     # Ramsey-type error during a gate
        g_r = s_rad * np.sqrt(A * abs(np.log(TWO_PI * t_gate_ns * 1e-9)))
        out["gate_error"] = float((t_gate_ns * 1e-9 * g_r) ** 2 / 3)
    return out


# ---------------------------------------------------------------- 2. DRAG single-qubit gate
def _ops(d):
    a = np.diag(np.sqrt(np.arange(1, d)), 1)
    return a, a.T, np.diag(np.arange(d, dtype=float))


def sx_propagator(x, T=25.0, d=3, dt=0.1, drag=True):
    """Gaussian pulse in the rotating frame. x = [A, lam, delta, za, zb] (rad/ns); returns U."""
    A, lam, delta = x[0], (x[1] if drag else 0.0), (x[2] if drag else 0.0)
    a, ad, n = _ops(d)
    al = TWO_PI * ALPHA * 1e-3
    H0 = (al / 2) * (n @ n - n) + delta * n
    sig = T / 4; g0 = np.exp(-(T / 2) ** 2 / (2 * sig ** 2))
    U = np.eye(d, dtype=complex)
    for k in range(int(round(T / dt))):
        t = (k + 0.5) * dt
        e = np.exp(-(t - T / 2) ** 2 / (2 * sig ** 2))
        g = (e - g0) / (1 - g0); dg = -(t - T / 2) / sig ** 2 * e / (1 - g0)
        ox, oy = A * g, -lam * A * dg / al
        H = H0 + ox * (a + ad) / 2 + oy * 1j * (ad - a) / 2
        U = expm(-1j * H * dt) @ U
    return U


def _sx_fidelity(x, T, d, drag):
    U = sx_propagator(x, T, d, drag=drag)[:2, :2]
    za, zb = x[-2], x[-1]
    Rz = lambda p: np.diag([np.exp(-1j * p / 2), np.exp(1j * p / 2)])
    V = Rz(za) @ (np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)) @ Rz(zb)
    M = V.conj().T @ U
    return (np.trace(M.conj().T @ M).real + abs(np.trace(M)) ** 2) / 6     # Pedersen, includes leakage


def optimise_sx(T=25.0, d=3, drag=True):
    x0 = [0.10, 0.5, 0.0, 0.0, 0.0] if drag else [0.10, 0.0, 0.0, 0.0, 0.0]
    obj = (lambda x: 1 - _sx_fidelity(x, T, d, True)) if drag else \
          (lambda x: 1 - _sx_fidelity([x[0], 0, 0, x[1], x[2]], T, d, False))
    start = x0 if drag else [x0[0], x0[3], x0[4]]
    r = minimize(obj, start, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-13, "maxiter": 4000})
    x = r.x if drag else [r.x[0], 0, 0, r.x[1], r.x[2]]
    return x, 1 - _sx_fidelity(x, T, d, drag)


def part_drag():
    T0 = float(B("t_1q_ns"))
    print(f"DRAG single-qubit gate (Gaussian, α={ALPHA} MHz, 3-level optimisation, re-checked at 4 levels)")
    rows = {}
    for T in (8.0, 10.0, 15.0, T0):
        _, e0 = optimise_sx(T, 3, False)
        x1, e1 = optimise_sx(T, 3, True)
        e1_4 = 1 - _sx_fidelity(x1, T, 4, True)
        rows[f"{T:g}"] = {"no_drag": float(e0), "drag": float(e1), "drag_4level": float(e1_4)}
        print(f" {T:>4g} ns: no DRAG {e0:.1e} | DRAG {e1:.1e} | DRAG pulse at 4 levels {e1_4:.1e}")
    print(" (values below ~1e-8 are at the numerical floor of this model; no-DRAG is not monotonic in T because phase errors interfere)")
    r25, r10 = rows[f"{T0:g}"], rows["10"]
    print(f" at {T0:g} ns a plain Gaussian already has error < 1e-4 (DRAG not needed): {ok(r25['no_drag'] < 1e-4)}")
    print(f" at 10 ns DRAG is > 100x better than a plain Gaussian and 4-level error < 1e-5: "
          f"{ok(r10['drag'] < r10['no_drag'] / 100 and r10['drag_4level'] < 1e-5)}")
    return {"time_ns": T0, "sweep": rows, "error_used_in_budget": float(max(r25["drag_4level"], 1e-9))}


# ---------------------------------------------------------------- 3. readout and Purcell
def readout(fq, fr, T1_us=None):
    g, al = B("g_qr_MHz") * 1e-3, ALPHA * 1e-3                       # GHz
    D = fq - fr
    chi = g ** 2 / D * al / (D + al)                                 # GHz; resonator shifts by +-chi
    kappa = 2 * abs(chi)                                             # SNR-optimal kappa = 2 chi
    ncrit = D ** 2 / (4 * g ** 2)
    nbar = B("nbar_frac") * ncrit
    k_rad, chi_rad = kappa * TWO_PI * 1e9, abs(chi) * TWO_PI * 1e9
    dalpha2 = 4 * chi_rad ** 2 * nbar / (k_rad ** 2 / 4 + chi_rad ** 2)
    tau = max(B("t_readout_ns") * 1e-9 - 2 / k_rad, 0)               # drop the ring-up time
    snr = np.sqrt(2 * B("eta") * k_rad * tau * dalpha2)
    e_snr = 0.5 * erfc(snr / (2 * np.sqrt(2)))
    T1 = (T1_us or B("T1_us")) * 1e-6
    e_t1 = B("t_readout_ns") * 1e-9 / (2 * T1)
    gam = k_rad * (g / D) ** 2 * (fq / fr)                           # Purcell rate, no filter
    gam_f = gam * fr / (2 * B("Q_purcell") * abs(D))                 # with bandpass Purcell filter
    return {"chi_MHz": float(abs(chi) * 1e3), "kappa_MHz": float(kappa * 1e3), "n_crit": float(ncrit),
            "nbar": float(nbar), "snr": float(snr), "error": float(e_snr + e_t1),
            "error_snr": float(e_snr), "error_t1_decay": float(e_t1),
            "T1_purcell_nofilter_us": float(1e6 / gam), "T1_purcell_filter_us": float(1e6 / gam_f)}


def part_readout():
    print(f"Readout (g_qr={B('g_qr_MHz')} MHz, κ=2χ, η={B('eta')}, window {B('t_readout_ns')} ns, "
          f"n̄={B('nbar_frac')}·n_crit, Purcell filter Q={B('Q_purcell')})")
    res = {}
    for die in ("kahler_die", "hyperkahler_die"):
        for q in SPEC[die]["qubits"]:
            r = res[q["id"]] = readout(q["f_park"], q["readout_res"])
            print(f" {q['id']}: f_q={q['f_park']:.2f}, f_r={q['readout_res']:.2f} GHz | χ={r['chi_MHz']:.2f} MHz, "
                  f"κ={r['kappa_MHz']:.2f} MHz, n_crit={r['n_crit']:.0f} | SNR={r['snr']:.1f}, readout error "
                  f"{r['error']:.2e} | Purcell T1 {r['T1_purcell_nofilter_us']:.0f} µs bare, "
                  f"{r['T1_purcell_filter_us']:.0f} µs filtered")
        fr = sorted(q["readout_res"] for q in SPEC[die]["qubits"])
        sp = min(np.diff(fr)) if len(fr) > 1 else 9.9
        print(f"  {die} resonator spacing {1e3 * sp:.0f} MHz")
        res[die + "_min_spacing_MHz"] = float(1e3 * sp)
    rs = [v for k, v in res.items() if isinstance(v, dict)]
    print(f" readout error < 1% for every qubit: {ok(all(r['error'] < 0.01 for r in rs))}")
    print(f" Purcell filter needed (bare T1 limit < T1) and sufficient (> 5·T1): "
          f"{ok(all(r['T1_purcell_nofilter_us'] < B('T1_us') and r['T1_purcell_filter_us'] > 5 * B('T1_us') for r in rs))}")
    print(f" resonator spacing >= 100 MHz on both dies: "
          f"{ok(res['kahler_die_min_spacing_MHz'] >= 100 and res['hyperkahler_die_min_spacing_MHz'] >= 100)}")
    return res


# ---------------------------------------------------------------- flux noise table
def part_flux():
    d = B("squid_d")
    print(f"Flux response and 1/f noise (SQUID asymmetry d={d}, √A={B('A_phi_uPhi0')} µΦ0/√Hz)")
    pts = {}
    for die in ("kahler_die", "hyperkahler_die"):
        for q in SPEC[die]["qubits"]:
            pts[f"{q['id']}_park"] = (q["f_park"], q["f_max"])
    pts["K1_CZ"] = (SPEC["kahler_die"]["qubits"][0]["f_park"] + ALPHA * 1e-3, B("fmax_GHz"))
    for i in range(3):
        pts[f"H{i}_operate"] = (SPEC["hyperkahler_die"]["operating_freq"], SPEC["hyperkahler_die"]["qubits"][i]["f_max"])
    out = {}
    for name, (f, fm) in pts.items():
        out[name] = r = flux_noise(f, fm, d, t_gate_ns=35)
        print(f" {name:<11} f={f:.3f} GHz (fmax {fm:.2f}): Φ={r['flux_Phi0']:.3f} Φ0, slope {r['slope_GHz_per_Phi0']:.2f} GHz/Φ0, "
              f"T_φ,echo {r['Tphi_echo_us']:.0f} µs, quasi-static σ_f {1e3 * r['sigma_f_quasistatic_MHz']:.0f} kHz")
    hk_sig = max(v["sigma_f_quasistatic_MHz"] for k, v in out.items() if k.endswith("operate"))
    print(f" flux-noise jitter at the hyperkähler operating point < 0.1·su2_tol ({0.1 * B('su2_tol_MHz')} MHz): "
          f"{ok(hk_sig < 0.1 * B('su2_tol_MHz'))}")
    slope = max(v["slope_GHz_per_Phi0"] for k, v in out.items() if k.endswith("operate"))
    bits = int(np.ceil(np.log2(1.0 / (0.1 * B("su2_tol_MHz") * 1e-3 / slope))))
    print(f" flux DAC for 0.1·tol frequency steps over a 1 Φ0 span: {bits} bits (needs <= 16): {ok(bits <= 16)}")
    reach = all(squid_freq(0.5, fm, d) < f <= fm - 0.01 for f, fm in pts.values())
    print(f" every operating/parking point inside its SQUID tuning range with 10 MHz margin: {ok(reach)}")
    out["dac_bits"] = bits
    return out


# ---------------------------------------------------------------- 4. error budget
def schedule_duration(tc, target):
    """ASAP duration in ns (per-qubit clocks)."""
    clock = {}
    for inst in tc.data:
        qs = [tc.find_bit(q).index for q in inst.qubits]
        if inst.operation.name in ("barrier", "measure"):
            continue
        dur = (target[inst.operation.name][tuple(qs)].duration or 0) * 1e9
        t0 = max(clock.get(q, 0) for q in qs)
        for q in qs:
            clock[q] = t0 + dur
    return max(clock.values())


def part_budget(drag, flux, ro_res):
    print("Per-gate error budget (T1/T2 coherence + coherent error + flux noise during the gate)")
    sc = lambda t: (t / 35) ** 2                                    # flux error grows ~ t^2
    q1 = {"kahler_die": coherence_error(B("t_1q_ns")) + drag["error_used_in_budget"] + flux["K0_park"]["gate_error"] * sc(B("t_1q_ns")),
          "hyperkahler_die": coherence_error(B("t_1q_ns")) + drag["error_used_in_budget"] + flux["H0_park"]["gate_error"] * sc(B("t_1q_ns"))}
    with contextlib.redirect_stdout(io.StringIO()):
        czk = cs.part_cz(g=B("g_cz_eff_MHz"), verbose=False)
        czh = cs.part_cz(g=B("g_tri_MHz"), verbose=False)
    cz_k = czk["lindblad_error"] + flux["K1_CZ"]["gate_error"] * sc(czk["time_ns"])
    cz_h = czh["lindblad_error"] + 2 * flux["H0_operate"]["gate_error"] * sc(czh["time_ns"])
    t, leak, phi, F = cs.exchange_gate(B("g_tri_MHz"), ALPHA, 3)
    ex_h = (1 - F) + coherence_error(t, 2) + 2 * flux["H0_operate"]["gate_error"] * sc(t)
    ro = max(v["error"] for v in ro_res.values() if isinstance(v, dict))
    tab = {"kahler_die": {"1q": q1["kahler_die"], "cz": cz_k, "cz_time_ns": czk["time_ns"]},
           "hyperkahler_die": {"1q": q1["hyperkahler_die"], "cz": cz_h, "cz_time_ns": czh["time_ns"],
                               "xy_exchange": ex_h, "xy_exchange_time_ns": t,
                               "xy_exchange_coherent_error": 1 - F}, "worst_readout_error": ro}
    print(f" Kähler die      : 1Q {q1['kahler_die']:.2e} | CZ ({czk['time_ns']:.0f} ns) {cz_k:.2e}")
    print(f" Hyperkähler die : 1Q {q1['hyperkahler_die']:.2e} | CZ with g={B('g_tri_MHz')} MHz ({czh['time_ns']:.0f} ns) {cz_h:.2e} | "
          f"XY exchange ({t:.0f} ns) {ex_h:.2e} (coherent part {1 - F:.2e})")
    print(f" worst readout error {ro:.2e}")
    print(f" targets met (1Q < 5e-4, Kähler CZ < 5e-3, exchange < 1e-2): "
          f"{ok(max(q1.values()) < 5e-4 and cz_k < 5e-3 and ex_h < 1e-2)}")
    print(f" fixed-g triangle CZ is worse than the tunable-coupler Kähler CZ: {ok(cz_h > cz_k)}")
    return tab


def part_circuit(tab, flux):
    print("Hyperkähler unit compiled with the Stage 1 errors, plus idle flux dephasing")
    h = tab["hyperkahler_die"]
    ng = {"rz": {"duration": 0, "error": 0},
          "sx": {"duration": B("t_1q_ns") * 1e-9, "error": h["1q"]}, "x": {"duration": B("t_1q_ns") * 1e-9, "error": h["1q"]},
          "cz": {"duration": h["cz_time_ns"] * 1e-9, "error": h["cz"]}}
    old = SPEC["native_gates"]; SPEC["native_gates"] = ng
    tgt = cs.die_target(SPEC["hyperkahler_die"]); SPEC["native_gates"] = old
    T2f = np.mean([flux[f"H{i}_operate"]["Tphi_echo_us"] for i in range(3)])
    res = {}
    for steps in (1, 5):
        tc = transpile(hk.circuit(0.8, steps=steps), target=tgt, optimization_level=3, seed_transpiler=7)
        f_gates = cs.est_fidelity(tc, tgt)
        dur = schedule_duration(tc, tgt)
        c = np.exp(-(dur * 1e-3 / T2f) ** 2)                          # echo-quality refocusing assumed
        f_idle = ((2 + c) / 3) ** 3                                   # 3 qubits
        f_nat = (1 - h["xy_exchange"] - h["cz"]) ** (3 * steps)       # XY exchange + one CPhase per pair per step
        res[str(steps)] = {"cz_count": int(tc.count_ops().get("cz", 0)), "duration_ns": float(dur),
                           "fidelity_gates": float(f_gates), "fidelity_with_flux_idle": float(f_gates * f_idle),
                           "native_xy_plus_cphase_estimate": float(f_nat)}
        print(f" {steps} step(s): {tc.count_ops().get('cz', 0)} CZ, duration {dur:.0f} ns | gate-error fidelity {f_gates:.3f} | "
              f"x idle flux dephasing {f_idle:.3f} -> {f_gates * f_idle:.3f} | native XY+CPhase estimate {f_nat:.3f} (ignores 1Q gates)")
    print(f" mean flux-limited echo T_φ at the operating point {T2f:.0f} µs (vs baseline T2 {B('T2_us')} µs): "
          f"{ok(T2f < B('T2_us'))}")
    return res


if __name__ == "__main__":
    flux = part_flux()
    drag = part_drag()
    ro = part_readout()
    tab = part_budget(drag, flux, ro)
    comp = part_circuit(tab, flux)
    json.dump({"decisions": cs.DECISIONS, "flux": flux, "drag": drag, "readout": ro,
               "budget": tab, "hk_compiled": comp}, open("stage1_budget.json", "w"), indent=2)
    print("wrote stage1_budget.json")
