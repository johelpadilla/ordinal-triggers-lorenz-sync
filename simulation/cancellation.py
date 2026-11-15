"""Cancellation identity on a fixed edge set (D_E, Dbar_E, C_E, kappa_E).

Delta_e = tau_b^(e)(t) - B_e;  S+ = sum_{Delta>0} Delta;  S- = sum_{Delta<0} |Delta|;
Dbar_E = (S+ - S-)/|E|;  D_E = (S+ + S-)/|E|;  C_E = 2 min(S+, S-)/|E|;
kappa_E = C_E / D_E (NaN if D_E = 0).
"""
import numpy as np


def kappa(c_e, d_e):
    c_e = np.asarray(c_e, float); d_e = np.asarray(d_e, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(d_e == 0, np.nan, c_e / np.where(d_e == 0, 1.0, d_e))
    return k if k.ndim else float(k)


def cancellation(delta):
    d = np.asarray(delta, float); n = d.shape[-1]
    sp = np.where(d > 0, d, 0).sum(-1); sm = np.where(d < 0, -d, 0).sum(-1)
    DE = (sp + sm) / n; DB = (sp - sm) / n; CE = 2 * np.minimum(sp, sm) / n
    return DE, DB, CE, kappa(CE, DE)
