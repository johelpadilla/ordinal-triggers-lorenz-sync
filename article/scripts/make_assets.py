#!/usr/bin/env python3
"""Builds the figures (vector PDF), LaTeX tables and number macros of the article
from the simulation outputs in ../../results. No simulation is re-run."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ART = HERE.parent
RES = ART.parent / "results"
FIG = ART / "figures"; GEN = ART / "generated"
r = pd.read_csv(RES / "runs.csv"); s = pd.read_csv(RES / "summary.csv")
t = pd.read_csv(RES / "tests.csv"); d = pd.read_csv(RES / "periodic_diagnostic.csv")
SC = ["clean", "noise", "distortion", "drift"]
SCN = {"clean": "Clean", "noise": "Noise", "distortion": "Distortion", "drift": "Drift"}
M = ["A", "B", "C", "R1", "R2", "R3a", "R3b", "V", "VA"]
RR = ["R1", "R2", "R3a", "R3b", "V"]
H = 0.01; W = 20

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = {"A": "#7f7f7f", "B": "#1f4e79", "C": "#8c564b", "R1": "#d62728", "R2": "#ff7f0e",
       "R3a": "#9467bd", "R3b": "#e377c2", "V": "#2ca02c", "VA": "#17becf", "P": "#000000"}
MK = {"A": "s", "B": "D", "C": "v", "R1": "o", "R2": "^", "R3a": "P", "R3b": "X", "V": "h", "VA": "*"}

def get(sc, m, k):
    return r[(r.scenario == sc) & (r.method == m)].sort_values("seed")[k].to_numpy(float)

# ---------------- Figure: MIET vs guaranteed bound
fig, ax = plt.subplots(figsize=(6.6, 2.9))
rng = np.random.default_rng(0)
order = ["B", "C"] + RR
gap = len(order) + 1.2
for i, sc in enumerate(SC):
    for j, m in enumerate(order):
        v = get(sc, m, "min_iet"); v = v[np.isfinite(v)] / H
        x = i * gap + j + rng.uniform(-0.22, 0.22, v.size)
        ax.scatter(x, v, s=9, marker=MK[m], facecolor="none" if m in ("B", "C") else COL[m],
                   edgecolor=COL[m], linewidths=0.6, zorder=3)
        nmiss = 20 - v.size
        if nmiss:
            ax.text(i * gap + j, 0.62, f"{nmiss}$^\\dagger$", ha="center", va="bottom", fontsize=6.3, color=COL[m])
    if i:
        ax.axvline(i * gap - 1.1, color="0.85", lw=0.5)
ax.axhline(2, color="k", ls="--", lw=0.8, zorder=2)
ax.set_yscale("log"); ax.set_ylim(0.55, 300)
ax.set_xticks([i * gap + (len(order) - 1) / 2 for i in range(len(SC))]); ax.set_xticklabels([SCN[x] for x in SC])
ax.set_xlim(-1, len(SC) * gap - 1.4)
ax.set_ylabel(r"minimum inter-event time per run $/\,h$")
ax.tick_params(axis="x", length=0)
ax.grid(axis="y", which="major", color="0.9", lw=0.5, zorder=0)
h_ = [Line2D([], [], ls="", marker=MK[m], markersize=4, markerfacecolor="none" if m in ("B", "C") else COL[m],
             markeredgecolor=COL[m], label=m) for m in order]
h_.append(Line2D([], [], color="k", ls="--", lw=0.8, label=r"bound $2h$"))
ax.legend(handles=h_, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False, handletextpad=0.2, columnspacing=1.0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.savefig(FIG / "fig_miet_bound.pdf", metadata={"CreationDate": None}); plt.close(fig)

# ---------------- Figure: RMS and updates
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.75))
x = np.arange(len(SC)); off = np.linspace(-0.36, 0.36, len(M))
for k, m in enumerate(M):
    mu = [s[(s.scenario == sc) & (s.method == m)].iloc[0] for sc in SC]
    a1.errorbar(x + off[k], [q.rms_mean for q in mu], yerr=[q.rms_sd for q in mu], fmt=MK[m], ms=3.6,
                color=COL[m], mfc=COL[m], mec=COL[m], elinewidth=0.6, capsize=1.5, label=m)
    a2.scatter(x + off[k], [q.updates_mean for q in mu], marker=MK[m], s=14, color=COL[m])
for a in (a1, a2):
    a.set_xticks(x); a.set_xticklabels([SCN[q] for q in SC]); a.set_yscale("log")
    a.grid(axis="y", which="major", color="0.9", lw=0.5); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for b in x[:-1]: a.axvline(b + 0.5, color="0.88", lw=0.5)
a1.set_ylabel(r"true RMS error, $t\geq 6$"); a2.set_ylabel("number of control updates on $[0,20]$")
a1.set_title("(a) Tracking error (mean $\\pm$ SD)", loc="left"); a2.set_title("(b) Update count (mean)", loc="left")
fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=9, frameon=False, bbox_to_anchor=(0.5, -0.06), handletextpad=0.1, columnspacing=0.9)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIG / "fig_rms_updates.pdf", metadata={"CreationDate": None}); plt.close(fig)

# ---------------- Figure: periodic diagnostic and V invariance
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8))
lo, hi = 0, 22
for sc, mk in zip(SC, ["o", "s", "^", "D"]):
    p_ = d[d.scenario == sc].sort_values("seed").rms.to_numpy(); r1 = get(sc, "R1", "rms")
    a1.scatter(p_, r1, s=11, marker=mk, facecolor="none", edgecolor=plt.cm.viridis(SC.index(sc) / 3.5), lw=0.7, label=SCN[sc])
a1.plot([lo, hi], [lo, hi], "k--", lw=0.6); a1.set_xlim(lo, hi); a1.set_ylim(lo, hi); a1.set_aspect("equal")
a1.set_xlabel("RMS, periodic schedule P (matched budget)"); a1.set_ylabel("RMS, event rule R1")
a1.set_title("(a) Timing diagnostic", loc="left"); a1.legend(frameon=False, loc="lower right", handletextpad=0.2)
vc, vd = get("clean", "V", "rms"), get("distortion", "V", "rms")
bc, bd = get("clean", "B", "rms"), get("distortion", "B", "rms")
a2.scatter(vc, vd, s=12, marker="h", color=COL["V"], label="V (R1 + rank law)")
a2.scatter(bc, bd, s=12, marker="D", facecolor="none", edgecolor=COL["B"], lw=0.7, label="B (dynamic magnitude)")
a2.plot([0, 26], [0, 26], "k--", lw=0.6); a2.set_xlim(0, 26); a2.set_ylim(0, 26); a2.set_aspect("equal")
a2.set_xlabel("RMS, clean sensor"); a2.set_ylabel(r"RMS, distorted sensor $\varphi(x)=x+10\tanh(x/15)$")
a2.set_title("(b) Monotone distortion, per seed", loc="left"); a2.legend(frameon=False, loc="lower right", handletextpad=0.2)
for a in (a1, a2):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig_diagnostics.pdf", metadata={"CreationDate": None}); plt.close(fig)

# ---------------- Tables
def pm(a, b, nd=2): return f"${a:.{nd}f}\\pm{b:.{nd}f}$"
NAME = {"A": "A (continuous)", "B": "B (dynamic magnitude)", "C": "C (naive ordinal)", "R1": "R1 (re-anchoring)",
        "R2": "R2 (hysteresis)", "R3a": "R3a (R1 + $\\kappa_E$)", "R3b": "R3b (R2 + $\\kappa_E$)",
        "V": "V (R1 + rank law)", "VA": "VA (rank law, every step)"}
L = []
for m in M:
    row = [NAME[m]]
    for sc in SC:
        q = s[(s.scenario == sc) & (s.method == m)].iloc[0]
        best = s[(s.scenario == sc) & (s.method != "A")].rms_mean.min() == q.rms_mean
        cell = pm(q.rms_mean, q.rms_sd)
        row += [("{\\boldmath" + cell + "}") if best else cell, f"{q.updates_mean:.1f}"]
    L.append(" & ".join(row) + r" \\")
    if m in ("C",): L.append(r"\midrule")
    if m == "R3b": L.append(r"\midrule")
(GEN / "tab_rms.tex").write_text("\n".join(L) + "\n")

L = []
for m in RR:
    for sc in SC:
        q = s[(s.scenario == sc) & (s.method == m)].iloc[0]
        mi = "---" if not np.isfinite(q.min_iet_min) else f"{q.min_iet_min:.2f}"
        me = "---" if not np.isfinite(q.med_iet_median) else f"{q.med_iet_median:.2f}"
        rr = r[(r.scenario == sc) & (r.method == m)]
        viol = int((rr.bound_ok.astype(str) == "0").sum())
        L.append(f"{NAME[m] if sc == 'clean' else ''} & {SCN[sc]} & {q.bound:.2f} & {mi} & {me} & "
                 f"{int(q.bound_ok_runs)} & {int(q.runs_no_IET)} & {viol} & {q.max_step_dtau:.2f} \\\\")
    if m != RR[-1]: L.append(r"\addlinespace[2pt]")
(GEN / "tab_miet.tex").write_text("\n".join(L) + "\n")

L = []
for sc in SC:
    p_ = d[d.scenario == sc].sort_values("seed"); a = p_.rms.to_numpy(); b = get(sc, "R1", "rms")
    pv = wilcoxon(b, a).pvalue
    L.append(f"{SCN[sc]} & {int(p_.period_steps.iloc[0])} & {p_.n_updates.mean():.0f} & {get(sc,'R1','n_updates').mean():.1f} & "
             f"{pm(a.mean(), a.std(ddof=1))} & {pm(b.mean(), b.std(ddof=1))} & {int((b < a).sum())}/20 & "
             + (f"${pv:.1e}".replace("e-0", r"\times10^{-") + "}$") + r" \\")
(GEN / "tab_periodic.tex").write_text("\n".join(L) + "\n")

# ---------------- Macros (every number quoted in the text)
def q(sc, m): return s[(s.scenario == sc) & (s.method == m)].iloc[0]
R = r[r.method.isin(RR)]
mac = {}
mac["nRuleRuns"] = len(R); mac["nMeasurable"] = int((R.bound_ok.astype(str) == "1").sum())
mac["nViol"] = int((R.bound_ok.astype(str) == "0").sum()); mac["nNoIET"] = int((R.bound_ok.astype(str) == "no_IET").sum())
mac["minIETglobal"] = f"{R.min_iet.min():.2f}"; mac["maxDtau"] = f"{R.max_step_dtau.max():.2f}"
mac["nTies"] = int(R.n_eval_ties.sum()); mac["nTauNaN"] = int(R.n_eval_nan.sum()); mac["nDiv"] = int(r.diverged.sum())
mac["nRmsNaN"] = int(r.rms.isna().sum()); mac["nRunsTotal"] = len(r)
mac["nRthreeaNoIET"] = int((r[r.method == "R3a"].bound_ok.astype(str) == "no_IET").sum())
mac["nRthreebNoIET"] = int((r[r.method == "R3b"].bound_ok.astype(str) == "no_IET").sum())
rr = s[s.method.isin(RR)]
mac["medIETmin"] = f"{rr.med_iet_median.min():.2f}"; mac["medIETmax"] = f"{rr.med_iet_median.max():.2f}"
pw = t.p_wilcoxon.unique(); ph = t.p_holm.unique(); assert len(pw) == 1 and len(ph) == 1
mac["pW"] = f"{pw[0]:.1e}".replace("e-06", r"\times10^{-6}"); mac["pHolm"] = f"{ph[0]:.1e}".replace("e-06", r"\times10^{-6}")
mac["maxWinsB"] = int(t[t.ref == "B"].wins.max()); mac["maxWinsC"] = int(t[t.ref == "C"].wins.max()); mac["nTests"] = len(t)
for sc, tag in zip(SC, ["Clean", "Noise", "Dist", "Drift"]):
    for m in M:
        mm = m.replace("3a", "threea").replace("3b", "threeb").replace("1", "one").replace("2", "two")
        mac[f"rms{mm}{tag}"] = f"{q(sc,m).rms_mean:.2f}"; mac[f"upd{mm}{tag}"] = f"{q(sc,m).updates_mean:.1f}"
    p_ = d[d.scenario == sc]; mac[f"rmsP{tag}"] = f"{p_.rms.mean():.2f}"; mac[f"winsRoneP{tag}"] = int((get(sc, 'R1', 'rms') < p_.sort_values('seed').rms.to_numpy()).sum())
    mac[f"updP{tag}"] = f"{p_.n_updates.mean():.0f}"
vb = get("distortion", "V", "rms"); bb = get("distortion", "B", "rms")
mac["pHfour"] = f"{wilcoxon(vb, bb, alternative='less').pvalue:.2f}"
mac["VidentRMS"] = int(np.sum(get("clean", "V", "rms") == get("distortion", "V", "rms")))
mac["VidentUpd"] = int(np.sum(get("clean", "V", "n_updates") == get("distortion", "V", "n_updates")))
# B vs C paired, from the per-seed data
for sc, tag in zip(SC, ["Clean", "Noise", "Dist", "Drift"]):
    b_, c_ = get(sc, "B", "rms"), get(sc, "C", "rms")
    mac[f"winsBC{tag}"] = int((b_ < c_).sum()); mac[f"pBC{tag}"] = f"{wilcoxon(b_, c_).pvalue:.3f}"
out = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
(GEN / "numbers.tex").write_text("% generated by scripts/make_assets.py from ../results/*.csv\n" + "\n".join(out) + "\n")
print("\n".join(out))
