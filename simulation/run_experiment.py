#!/usr/bin/env python3
"""Ordinal triggers R1-R3 (and a rank-based correction layer) in the two-layer
master-slave Lorenz bench. All data are produced by numerical simulation.

Policies A, B, C are run with `bench.simulate`. The new policies use
`simulate_new`, which replicates the same loop; its condition "A" is checked
bit for bit against `bench.simulate` as a control of the re-implementation.
Outputs are written to ../results.
"""
from __future__ import annotations
import csv, math, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench as T
from cancellation import cancellation

ROOT = HERE.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
REFERENCE = ROOT / "data" / "reference_results_v1.csv"

# ------------------------------------------------ fixed parameters
W = T.W                     # 20, as for C
H = T.DT                    # h = 0.01; p = 1 (evaluation at every sample)
THETA = 8.0 / W             # 0.40
DELTA = THETA / 2.0         # 0.20
THETA_K = 0.5
U_MAX = T.U_MAX             # 40
CAL_SEEDS = [100, 101, 102, 103, 104]
SEEDS = T.SEEDS             # 0..19
SCEN = T.SCENARIOS
OLD = ("A", "B", "C")
NEW = ("R1", "R2", "R3a", "R3b", "V", "VA")
METHODS = OLD + NEW
P_PERIOD = [10]  # set per scenario in main (periodic diagnostic)
BOUND = {"R1": math.ceil(THETA * W / 4 - 1e-12) * H,
         "R3a": math.ceil(THETA * W / 4 - 1e-12) * H,
         "R2": 2 * math.ceil(DELTA * W / 4 - 1e-12) * H,
         "R3b": 2 * math.ceil(DELTA * W / 4 - 1e-12) * H,
         "V": math.ceil(THETA * W / 4 - 1e-12) * H}
LABEL = {"A": "continuous", "B": "dynamic magnitude (Girard)", "C": "naive ordinal level",
         "R1": "R1 re-anchoring", "R2": "R2 hysteresis", "R3a": "R3 over R1 (kappa>=0.5)",
         "R3b": "R3 over R2 (kappa>=0.5)", "V": "R1 + rank correction", "VA": "rank correction every step",
         "P": "periodic (diagnostic)"}


def tau_b_ties(x, y):
    """tau_b (identical to bench.tau_b_fast) and whether the window has any tie."""
    n = x.size
    iu = np.triu_indices(n, 1)
    sx = np.sign((x[:, None] - x[None, :])[iu]); sy = np.sign((y[:, None] - y[None, :])[iu])
    p = float(np.dot(sx, sy)); n0 = n * (n - 1) // 2
    tx = int(np.sum(sx == 0)); ty = int(np.sum(sy == 0))
    den = np.sqrt((n0 - tx) * (n0 - ty))
    return (np.nan if den == 0.0 else p / den), (tx + ty) > 0


def taus3(hm, hs):
    out = np.empty(3); tie = False
    for i in range(3):
        out[i], ti = tau_b_ties(hm[:, i], hs[:, i]); tie = tie or ti
    return out, tie


def rank_last(win):
    """Mid-rank of the last value of the window, scaled to [0, 1]."""
    n = win.size
    if n < 2:
        return 0.0
    v = win[-1]
    return (np.sum(win < v) + 0.5 * (np.sum(win == v) - 1)) / (n - 1)


def rank_error(hm, hs, nh):
    return np.array([rank_last(hs[:nh, i]) - rank_last(hm[:nh, i]) for i in range(3)])


def simulate_new(seed, scenario, cond, B_cal=None):
    rng = np.random.default_rng(1000 + int(seed))
    xm = rng.uniform([-10.0, -10.0, 10.0], [10.0, 10.0, 30.0])
    xs = rng.uniform([-10.0, -10.0, 10.0], [10.0, 10.0, 30.0])
    noise = np.random.default_rng(50_000 + int(seed)).normal(0.0, T.NOISE_STD, size=(T.N_STEPS + 1, 3))
    rank_law = cond in ("V", "VA")
    t = 0.0; eta = float(T.ETA0)
    ys = T.measure(xs, t, scenario, noise[0]); e_meas = ys - xm
    hm = np.zeros((W, 3)); hs = np.zeros((W, 3)); hm[0] = xm; hs[0] = ys; nh = 1
    if rank_law:
        u = -U_MAX * rank_error(hm, hs, nh)
    else:
        u, s = T.compute_u(e_meas, eta); eta = T.update_eta(eta, s)
    events = [0]
    err = np.empty(T.N_STEPS + 1); err[0] = float(np.linalg.norm(xs - xm))
    diverged = False
    anchor = None                      # R1/R3a/V: anchor B_e
    armed = True                       # R2/R3b
    n_eval = n_nan = n_tie = 0; max_dtau = 0.0; prev = None
    tau_cal = []
    ev_taus = []   # (k, tau vector or None) at each evaluation, for the independent replay
    for k in range(1, T.N_STEPS + 1):
        t = k * T.DT
        d = T.disturbance(t)
        xm = T.rk4_step(xm, T.DT, T.SIGMA_M, T.RHO_M, T.BETA_M, np.zeros(3))
        xs = T.rk4_step(xs, T.DT, T.SIGMA_S, T.RHO_S, T.BETA_S, u + d)
        if not np.all(np.isfinite(xs)) or np.linalg.norm(xs) > 1e3 or np.linalg.norm(xm) > 1e3:
            diverged = True; err[k:] = np.nan; break
        ys = T.measure(xs, t, scenario, noise[k]); e_meas = ys - xm
        err[k] = float(np.linalg.norm(xs - xm))
        if nh < W:
            hm[nh] = xm; hs[nh] = ys; nh += 1
        else:
            hm[:-1] = hm[1:]; hs[:-1] = hs[1:]; hm[-1] = xm; hs[-1] = ys
        fire = False
        if cond in ("A", "VA"):
            fire = True
            if cond == "A" and B_cal == "record" and nh >= W and t >= T.T_TRANS - 1e-12:
                tv, _ = taus3(hm, hs)
                if np.all(np.isfinite(tv)):
                    tau_cal.append(tv)
        elif cond == "P":
            fire = (k % P_PERIOD[0] == 0)
        elif nh >= W:                    # p = 1: evaluation at every sample once the window is full
            tv, tie = taus3(hm, hs); n_eval += 1; n_tie += int(tie)
            ev_taus.append((k, tv.copy() if np.all(np.isfinite(tv)) else None))
            if not np.all(np.isfinite(tv)):
                n_nan += 1; prev = None
            else:
                if prev is not None and not tie and not prev[1]:
                    max_dtau = max(max_dtau, float(np.max(np.abs(tv - prev[0]))))
                prev = (tv, tie)
                if cond in ("R1", "R3a", "V"):
                    if anchor is None:
                        anchor = tv.copy()          # initial anchor, no event
                    DE, _, CE, kap = cancellation(tv - anchor)
                    ok = DE >= THETA
                    if cond == "R3a":
                        ok = ok and np.isfinite(kap) and kap >= THETA_K
                    if ok:
                        fire = True; anchor = tv.copy()   # re-anchoring B_e <- tau_b(t_k)
                else:                                     # R2 / R3b with calibration baseline
                    DE, _, CE, kap = cancellation(tv - B_cal)
                    if armed:
                        ok = DE >= THETA
                        if cond == "R3b":
                            ok = ok and np.isfinite(kap) and kap >= THETA_K
                        if ok:
                            fire = True; armed = False
                    elif DE <= THETA - DELTA:
                        armed = True
        if fire:
            if rank_law:
                u = -U_MAX * rank_error(hm, hs, nh)
            else:
                u, s = T.compute_u(e_meas, eta); eta = T.update_eta(eta, s)
            events.append(k)
    post = err[T.I_TRANS:]
    n_post_nan = int(np.sum(~np.isfinite(post)))
    rms = float(np.sqrt(np.mean(post ** 2))) if n_post_nan == 0 else np.nan
    iets = np.diff(np.asarray(events, float)) * T.DT
    row = dict(seed=int(seed), scenario=scenario, method=cond, rms=rms, n_updates=len(events),
               min_iet=float(iets.min()) if iets.size else np.nan,
               med_iet=float(np.median(iets)) if iets.size else np.nan,
               n_events_post=int(np.sum(np.asarray(events) >= T.I_TRANS)),
               diverged=int(diverged), n_post_nan=n_post_nan, n_eval=n_eval, n_eval_nan=n_nan,
               n_eval_ties=n_tie, max_step_dtau=max_dtau if n_eval else np.nan,
               bound=BOUND.get(cond, np.nan))
    if cond in BOUND:
        row["bound_ok"] = "no_IET" if not iets.size else int(row["min_iet"] >= row["bound"] - 1e-9)
    else:
        row["bound_ok"] = ""
    extra = dict(ev_taus=ev_taus, tau_cal=tau_cal, events=events)
    return row, extra


def replay(ev_taus, cond, B_fixed):
    """Independent check: re-derive the events from the tau_b series, with D_E and
    kappa computed here from the definition (without the cancellation module)."""
    def DE_kappa(delta):
        Sp = sum(d for d in delta if d > 0); Sm = sum(-d for d in delta if d < 0); n = len(delta)
        DE = (Sp + Sm) / n; CE = 2 * min(Sp, Sm) / n
        return DE, (CE / DE if DE > 0 else float("nan"))
    ev = [0]; anchor = None; armed = True
    for k, tv in ev_taus:
        if tv is None:
            continue
        if cond in ("R1", "R3a", "V"):
            if anchor is None:
                anchor = list(tv); continue
            DE, kap = DE_kappa([a - b for a, b in zip(tv, anchor)])
            if DE >= THETA and (cond != "R3a" or (kap == kap and kap >= THETA_K)):
                ev.append(k); anchor = list(tv)
        else:
            DE, kap = DE_kappa([a - b for a, b in zip(tv, B_fixed)])
            if armed:
                if DE >= THETA and (cond != "R3b" or (kap == kap and kap >= THETA_K)):
                    ev.append(k); armed = False
            elif DE <= THETA - DELTA:
                armed = True
    return ev


def run_old(seed, sc, cond):
    r = T.simulate(seed, sc, cond)
    return dict(seed=r["seed"], scenario=sc, method=cond, rms=r["rms"], n_updates=r["n_updates"],
                min_iet=r["min_iet"], med_iet=r["med_iet"], n_events_post=np.nan, diverged=r["diverged"],
                n_post_nan=0 if np.isfinite(r["rms"]) else -1, n_eval=np.nan, n_eval_nan=np.nan,
                n_eval_ties=np.nan, max_step_dtau=np.nan, bound=np.nan, bound_ok="")


FIELDS = ["seed", "scenario", "method", "rms", "n_updates", "min_iet", "med_iet", "n_events_post", "diverged",
          "n_post_nan", "n_eval", "n_eval_nan", "n_eval_ties", "max_step_dtau", "bound", "bound_ok"]


def fmt(x, nd=3):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj


def main():
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time(); log = []
    P = lambda s="": (log.append(s), print(s))
    P(f"W={W} h={H} theta={THETA} delta={DELTA} theta_kappa={THETA_K} bounds={BOUND}")

    # 0) self-checks of the definitions
    de, db, ce, kp = cancellation(np.array([0.3, -0.1, 0.0]))
    assert abs(de - 0.4 / 3) < 1e-12 and abs(db - 0.2 / 3) < 1e-12 and abs(ce - 0.2 / 3) < 1e-12 and abs(kp - 0.5) < 1e-12
    assert abs(de - (abs(db) + ce)) < 1e-12            # identity D_E = |Dbar_E| + C_E
    rng = np.random.default_rng(7)
    for _ in range(50):
        a, b = rng.normal(size=W), rng.normal(size=W)
        assert abs(tau_b_ties(a, b)[0] - T.tau_b_fast(a, b)) < 1e-15
        w = rng.normal(size=W)
        assert rank_last(w) == rank_last(w + 10 * np.tanh(w / 15))
    P("Self-checks (kappa, D_E = |Dbar_E| + C_E, tau_b, rank invariance): OK")

    # 1) calibration baseline B_e per scenario (policy A, seeds 100-104)
    B_cal = {}
    for sc in SCEN:
        taus = []
        for s in CAL_SEEDS:
            _, ex = simulate_new(s, sc, "A", B_cal="record"); taus += ex["tau_cal"]
        B_cal[sc] = np.mean(np.asarray(taus), axis=0)
        P(f"B_e calibration [{sc}] = {np.round(B_cal[sc], 4).tolist()} (windows = {len(taus)})")

    # 2) control of the re-implementation: new A == bench A, bit for bit
    rows = []; mism = 0
    for sc in SCEN:
        for s in SEEDS:
            ro = run_old(s, sc, "A"); rn, _ = simulate_new(s, sc, "A")
            if not (ro["rms"] == rn["rms"] and ro["n_updates"] == rn["n_updates"]):
                mism += 1
    P(f"Re-implemented loop vs bench (condition A, 80 runs): mismatches = {mism}")
    assert mism == 0

    # 3) main set of runs
    rep_bad = rep_n = 0
    for sc in SCEN:
        for s in SEEDS:
            for m in OLD:
                rows.append(run_old(s, sc, m))
            for m in NEW:
                r, ex = simulate_new(s, sc, m, B_cal=B_cal[sc]); rows.append(r)
                if m in BOUND:
                    rep = replay(ex["ev_taus"], m, list(B_cal[sc]))
                    rep_bad += int(rep != ex["events"])
                    rep_n += 1
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); [w.writerow(r) for r in rows]
    P(f"Independent replay of R1/R2/R3a/R3b/V from tau_b: {rep_n - rep_bad}/{rep_n} runs identical")
    assert rep_bad == 0

    # 4) A/B/C against the reference results of the earlier bench
    orig = {}
    with open(REFERENCE) as f:
        for r in csv.DictReader(f):
            orig[(int(r["seed"]), r["scenario"], r["condition"])] = r
    bad = 0
    for r in rows:
        if r["method"] in OLD:
            o = orig[(r["seed"], r["scenario"], r["method"])]
            if not (repr(float(o["rms"])) == repr(r["rms"]) and int(o["n_updates"]) == r["n_updates"]
                    and float(o["min_iet"]) == r["min_iet"] and float(o["med_iet"]) == r["med_iet"]):
                bad += 1
    P(f"A/B/C vs reference results (240 rows): mismatches = {bad}")

    get = lambda sc, m, k: np.array([r[k] for r in rows if r["scenario"] == sc and r["method"] == m], float)

    # 5) summary
    summ = []
    for sc in SCEN:
        for m in METHODS:
            rms = get(sc, m, "rms"); n = get(sc, m, "n_updates"); mi = get(sc, m, "min_iet"); me = get(sc, m, "med_iet")
            dv = get(sc, m, "diverged")
            rec = dict(scenario=sc, method=m, label=LABEL[m], rms_mean=np.nanmean(rms), rms_sd=np.nanstd(rms, ddof=1),
                       rms_median=np.nanmedian(rms), n_rms_finite=int(np.isfinite(rms).sum()),
                       updates_mean=n.mean(), updates_sd=n.std(ddof=1), min_iet_min=np.nanmin(mi),
                       min_iet_mean=np.nanmean(mi), med_iet_median=np.nanmedian(me), n_diverged=int(dv.sum()),
                       bound=BOUND.get(m, np.nan))
            if m in BOUND:
                rec["runs_no_IET"] = int(sum(r["bound_ok"] == "no_IET" for r in rows if r["scenario"] == sc and r["method"] == m))
                rec["bound_ok_runs"] = int(sum(r["bound_ok"] == 1 for r in rows if r["scenario"] == sc and r["method"] == m))
                rec["max_step_dtau"] = float(np.nanmax(get(sc, m, "max_step_dtau")))
                rec["eval_ties"] = int(np.nansum(get(sc, m, "n_eval_ties"))); rec["eval_nan"] = int(np.nansum(get(sc, m, "n_eval_nan")))
            summ.append(rec)
    keys = list(summ[0].keys()) + ["runs_no_IET", "bound_ok_runs", "max_step_dtau", "eval_ties", "eval_nan"]
    keys = list(dict.fromkeys(keys))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in summ]

    P("\n=== Summary (20 seeds; true RMS error for t >= 6; IET in time units) ===")
    P(f"{'scenario':<11}{'method':<7}{'RMS mean+-sd':>16}{'updates':>10}{'IETmin':>8}{'IETmed':>8}{'div':>5}{'bound':>7}{'ok':>5}{'max|dtau|':>10}{'ties':>6}")
    for r in summ:
        P(f"{r['scenario']:<11}{r['method']:<7}{fmt(r['rms_mean']):>9}+-{fmt(r['rms_sd']):<6}{r['updates_mean']:>9.1f}"
          f"{r['min_iet_min']:>8.2f}{r['med_iet_median']:>8.2f}{r['n_diverged']:>5}"
          f"{(fmt(r['bound'],2) if r['method'] in BOUND else ''):>7}{(str(r.get('bound_ok_runs',''))):>5}"
          f"{(fmt(r.get('max_step_dtau',np.nan),4) if r['method'] in BOUND else ''):>10}{str(r.get('eval_ties','')):>6}")

    # 6) paired tests
    tests = []
    for ref in ("B", "C"):
        for m in NEW:
            for sc in SCEN:
                a, b = get(sc, m, "rms"), get(sc, ref, "rms"); ok = np.isfinite(a) & np.isfinite(b)
                dlt = a[ok] - b[ok]
                p = 1.0 if np.allclose(dlt, 0) else float(wilcoxon(a[ok], b[ok], alternative="two-sided").pvalue)
                tests.append(dict(ref=ref, method=m, scenario=sc, metric="rms", n=int(ok.sum()),
                                  mean_diff=float(dlt.mean()), median_diff=float(np.median(dlt)),
                                  wins=int(np.sum(dlt < 0)), p_wilcoxon=p))
    for ref in ("B", "C"):          # Holm per method over the four scenarios
        for m in NEW:
            idx = [i for i, t_ in enumerate(tests) if t_["ref"] == ref and t_["method"] == m]
            adj = holm([tests[i]["p_wilcoxon"] for i in idx])
            for i, pa in zip(idx, adj):
                tests[i]["p_holm"] = float(pa)
    a, b = get("distortion", "V", "rms"), get("distortion", "B", "rms")
    h4 = float(wilcoxon(a, b, alternative="less").pvalue)
    vc, vd = get("clean", "V", "rms"), get("distortion", "V", "rms")
    nv = [r for r in rows if r["method"] == "V"]
    ident = bool(np.array_equal(vc, vd))
    ident_ev = all(
        (rc["n_updates"], rc["min_iet"], rc["med_iet"]) == (rd["n_updates"], rd["min_iet"], rd["med_iet"])
        for rc, rd in zip([r for r in nv if r["scenario"] == "clean"], [r for r in nv if r["scenario"] == "distortion"]))
    with open(OUT / "tests.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0].keys())); w.writeheader(); [w.writerow(t_) for t_ in tests]
    P("\n=== Two-sided paired Wilcoxon tests on RMS (diff = method - ref; 'wins' = seeds with lower RMS) ===")
    for t_ in tests:
        P(f"{t_['method']:<4} vs {t_['ref']}  {t_['scenario']:<11} mean diff={t_['mean_diff']:+.3f}  wins {t_['wins']:>2}/{t_['n']}  "
          f"p={t_['p_wilcoxon']:.2e}  p_Holm={t_['p_holm']:.2e}")
    P(f"\nH4: V(distortion) < B(distortion), one-sided Wilcoxon p = {h4:.3e}")
    P(f"H4: RMS of V identical clean vs distortion in all 20 seeds: {ident}; events identical: {ident_ev}")

    allb = [r for r in rows if r["method"] in BOUND]
    mx = float(np.nanmax([r['max_step_dtau'] for r in allb]))
    P(f"\nH1: of {len(allb)} rule runs: comply {sum(r['bound_ok'] == 1 for r in allb)}, "
      f"violate {sum(r['bound_ok'] == 0 for r in allb)}, no IET (<= 1 event) {sum(r['bound_ok'] == 'no_IET' for r in allb)}")
    P(f"    smallest IET under any rule = {min(r['min_iet'] for r in allb if np.isfinite(r['min_iet'])):.4f} (bound 0.02)")
    P(f"    max one-step |dtau_b| (tie-free windows) = {mx!r} (bound 4/W = {4/W}); <= bound: {mx <= 4/W + 1e-12}")
    P(f"    evaluated windows with ties: {sum(r['n_eval_ties'] for r in allb)}; undefined tau_b: {sum(r['n_eval_nan'] for r in allb)}")
    P(f"Total divergences: {sum(r['diverged'] for r in rows)}; rows with undefined RMS: {sum(not np.isfinite(r['rms']) for r in rows)}")

    # 7) diagnostic: periodic schedule P with about as many updates as R1
    P("\n=== Diagnostic: periodic schedule P with about as many updates as R1 ===")
    diag = []
    for sc in SCEN:
        nr1 = float(np.mean(get(sc, "R1", "n_updates")))
        P_PERIOD[0] = int(round(T.N_STEPS / (nr1 - 1)))
        for s in SEEDS:
            r, _ = simulate_new(s, sc, "P"); r["period_steps"] = P_PERIOD[0]; diag.append(r)
        a = np.array([r["rms"] for r in diag if r["scenario"] == sc]); b = get(sc, "R1", "rms")
        p = float(wilcoxon(b, a).pvalue)
        P(f"{sc:<11} period={P_PERIOD[0]} steps  updates={np.mean([r['n_updates'] for r in diag if r['scenario']==sc]):.0f}  "
          f"RMS P={a.mean():.3f}+-{a.std(ddof=1):.3f}  RMS R1={b.mean():.3f}  R1 better {int(np.sum(b < a))}/20  Wilcoxon p={p:.2e}")
    with open(OUT / "periodic_diagnostic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["period_steps"]); w.writeheader(); [w.writerow(r) for r in diag]

    P(f"\nTotal time: {time.time() - t0:.1f} s")
    (OUT / "run_summary.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
