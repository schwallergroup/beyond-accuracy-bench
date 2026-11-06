import numpy as np
from numba import jit

# --- M1 ---
M1_SPECIES_ORDER = ['P', 'S', 'cat', 'catS']
M1_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1']
@jit(nopython=True)
def hardcoded_m1_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2):
    p, s, cat, cats = concentrations
    # S + cat <=> catS (k1, k_minus_1)
    # catS <=> P + cat (k2, k_minus_2)
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt])

# --- M2 ---
M2_SPECIES_ORDER = ['P', 'S', 'cat', 'cat2', 'catS']
M2_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1', '(R6).k1']
@jit(nopython=True)
def hardcoded_m2_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k3, k_minus_3):
    p, s, cat, cat2, cats = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat - 2 * k3 * cat**2 + 2 * k_minus_3 * cat2
    dcat2_dt = k3 * cat**2 - k_minus_3 * cat2
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat
    return np.array([dp_dt, ds_dt, dcat_dt, dcat2_dt, dcats_dt])

# --- M3 ---
M3_SPECIES_ORDER = ['P', 'S', 'cat', 'cat2', '(cat)2S']
M3_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1', '(R6).k1']
@jit(nopython=True)
def hardcoded_m3_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k3, k_minus_3):
    p, s, cat, cat2, cat_2_S = concentrations
    # R1: S + cat2 <=> (cat)2S (k1, k_minus_1) -> cat_2_S
    # R2: (cat)2S <=> P + cat2 (k2, k_minus_2)
    # R3 (was R5 in create_models): 2 cat <=> cat2 (k3, k_minus_3)
    
    # Note: create_models defines:
    # R1: S + cat2 -> (cat)2S
    # R2: (cat)2S -> S + cat2
    # R3: (cat)2S -> P + cat2
    # R4: P + cat2 -> (cat)2S
    # R5: 2 cat -> cat2
    # R6: cat2 -> 2 cat
    
    # Mapping to hardcoded_ode params (k1, k_minus_1, k2, k_minus_2, k3, k_minus_3):
    # k1/k_minus_1 corresponds to S binding: S + cat2 <=> cat_2_S
    # k2/k_minus_2 corresponds to P release: cat_2_S <=> P + cat2
    # k3/k_minus_3 corresponds to Dimerization: 2 cat <=> cat2

    # Rates
    v1_fwd = k1 * s * cat2
    v1_rev = k_minus_1 * cat_2_S
    
    v2_fwd = k2 * cat_2_S
    v2_rev = k_minus_2 * p * cat2
    
    v3_fwd = k3 * cat**2
    v3_rev = k_minus_3 * cat2

    dp_dt = v2_fwd - v2_rev
    ds_dt = -v1_fwd + v1_rev
    
    dcat_dt = -2 * v3_fwd + 2 * v3_rev
    
    dcat2_dt = -v1_fwd + v1_rev + v2_fwd - v2_rev + v3_fwd - v3_rev
    
    dcat_2_S_dt = v1_fwd - v1_rev - v2_fwd + v2_rev

    return np.array([dp_dt, ds_dt, dcat_dt, dcat2_dt, dcat_2_S_dt])

# --- M4 ---
M4_SPECIES_ORDER = ['P', 'S', 'X', 'cat', 'catS']
M4_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1']
@jit(nopython=True)
def hardcoded_m4_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2):
    p, s, x, cat, cats = concentrations
    dp_dt = k2 * x * cats - k_minus_2 * p * cat
    ds_dt = k1 * x * cats - k_minus_1 * s * cat
    dx_dt = -k1 * x * cats + k_minus_1 * s * cat - k2 * x * cats + k_minus_2 * p * cat
    dcat_dt = k1 * x * cats - k_minus_1 * s * cat + k2 * x * cats - k_minus_2 * p * cat
    dcats_dt = -k1 * x * cats + k_minus_1 * s * cat - k2 * x * cats + k_minus_2 * p * cat
    return np.array([dp_dt, ds_dt, dx_dt, dcat_dt, dcats_dt])

# --- M5 ---
M5_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'catP']
M5_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1', '(R6).k1']
@jit(nopython=True)
def hardcoded_m5_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k3, k_minus_3):
    p, s, cat, cats, catp = concentrations
    v1_fwd, v1_rev = k1 * s * cat, k_minus_1 * cats
    v2_fwd, v2_rev = k2 * cats * cat, k_minus_2 * catp * cat
    v3_fwd, v3_rev = k3 * catp, k_minus_3 * p * cat
    dp_dt = v3_fwd - v3_rev
    ds_dt = -v1_fwd + v1_rev
    dcat_dt = -v1_fwd + v1_rev + v3_fwd - v3_rev
    dcats_dt = v1_fwd - v1_rev - v2_fwd + v2_rev
    dcatp_dt = v2_fwd - v2_rev - v3_fwd + v3_rev
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dcatp_dt])

# --- M6 ---
M6_SPECIES_ORDER = ['P', 'S', 'cat', 'cat*', 'cat*S']
M6_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m6_ode(t, concentrations, k_act_fwd, k1, k_minus_1, k2, k_minus_2):
    p, s, cat, cat_star, cat_star_s = concentrations
    dp_dt = k2 * cat_star_s - k_minus_2 * p * cat_star
    ds_dt = -k1 * s * cat_star + k_minus_1 * cat_star_s
    dcat_dt = -k_act_fwd * cat
    dcat_star_s_dt = k1 * s * cat_star - k_minus_1 * cat_star_s - k2 * cat_star_s + k_minus_2 * p * cat_star
    dcat_star_dt = k_act_fwd * cat - k1 * s * cat_star + k_minus_1 * cat_star_s + k2 * cat_star_s - k_minus_2 * p * cat_star
    return np.array([dp_dt, ds_dt, dcat_dt, dcat_star_dt, dcat_star_s_dt])

# --- M7 ---
M7_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'catS2']
M7_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1', '(R6).k1']
@jit(nopython=True)
def hardcoded_m7_ode(t, concentrations, k1, k_minus_1, k3, k_minus_3, k2, k_minus_2):
    p, s, cat, cats, cats2 = concentrations
    dp_dt = k2 * cats2 - k_minus_2 * p * cats
    ds_dt = -k1 * s * cat + k_minus_1 * cats - k3 * s * cats + k_minus_3 * cats2
    dcat_dt = -k1 * s * cat + k_minus_1 * cats
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k3 * s * cats + k_minus_3 * cats2 + k2 * cats2 - k_minus_2 * p * cats
    dcats2_dt = k3 * s * cats - k_minus_3 * cats2 - k2 * cats2 + k_minus_2 * p * cats
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dcats2_dt])

# --- M8 ---
M8_SPECIES_ORDER = ['L', 'P', 'S', 'cat', 'cat*S', 'cat*']
M8_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1', '(R6).k1']
@jit(nopython=True)
def hardcoded_m8_ode(t, concentrations,
                     k1, k_minus_1,
                     k2, k_minus_2,
                     k3, k_minus_3):

    l, p, s, cat, cat_star_s, cat_star = concentrations

    # reaction rates
    v1_fwd = k1 * s * cat_star
    v1_rev = k_minus_1 * cat_star_s

    v2_fwd = k2 * cat_star_s
    v2_rev = k_minus_2 * p * cat_star

    v3_fwd = k3 * cat
    v3_rev = k_minus_3 * cat_star * l

    # ODEs
    dp_dt = v2_fwd - v2_rev
    ds_dt = -v1_fwd + v1_rev

    dcat_star_s_dt = v1_fwd - v1_rev - v2_fwd + v2_rev

    dcat_star_dt = (
        -v1_fwd + v1_rev
        +v2_fwd - v2_rev
        +v3_fwd - v3_rev
    )

    dcat_dt = -v3_fwd + v3_rev
    dl_dt   =  v3_fwd - v3_rev

    return np.array([dl_dt, dp_dt, ds_dt, dcat_dt, dcat_star_s_dt, dcat_star_dt])

# --- M9 ---
M9_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_cat']
M9_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m9_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k3):
    p, s, cat, cats, inactive_cat = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat - k3 * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat
    dinactive_cat_dt = k3 * cat
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cat_dt])

# --- M10 ---
M10_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_catI', 'inhibitor']
M10_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m10_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k3):
    p, s, cat, cats, inactive_cati, inhibitor = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat - k3 * inhibitor * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat
    dinactive_cati_dt = k3 * inhibitor * cat
    dinhibitor_dt = -k3 * inhibitor * cat
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cati_dt, dinhibitor_dt])

# --- M11 ---
M11_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_catS']
M11_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m11_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k_deact_fwd):
    p, s, cat, cats, inactive_cats = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats - k_deact_fwd * s * cat
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat - k_deact_fwd * s * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat
    dinactive_cats_dt = k_deact_fwd * s * cat
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cats_dt])

# --- M12 ---
M12_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_catP']
M12_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m12_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k_deact_fwd):
    p, s, cat, cats, inactive_catp = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat - k_deact_fwd * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat - k_deact_fwd * p * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat
    dinactive_catp_dt = k_deact_fwd * p * cat
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_catp_dt])

# --- M13 ---
M13_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_cat2']
M13_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m13_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k_deact_fwd):
    p, s, cat, cats, inactive_cat2 = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat - 2 * k_deact_fwd * cat**2
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat
    dinactive_cat2_dt = k_deact_fwd * cat**2
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cat2_dt])

# --- M14 ---
M14_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_catS']
M14_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m14_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k_deact_fwd):
    p, s, cat, cats, inactive_cats = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat - k_deact_fwd * cats
    dinactive_cats_dt = k_deact_fwd * cats
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cats_dt])

# --- M15 ---
M15_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_catSI', 'inhibitor']
M15_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m15_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k_deact_fwd):
    p, s, cat, cats, inactive_catsi, inhibitor = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat - k_deact_fwd * inhibitor * cats
    dinactive_catsi_dt = k_deact_fwd * inhibitor * cats
    dinhibitor_dt = -k_deact_fwd * inhibitor * cats
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_catsi_dt, dinhibitor_dt])

# --- M16 ---
M16_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_catS2']
M16_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m16_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k_deact_fwd):
    p, s, cat, cats, inactive_cats2 = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats - k_deact_fwd * s * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat - k_deact_fwd * s * cats
    dinactive_cats2_dt = k_deact_fwd * s * cats
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cats2_dt])

# --- M17 ---
M17_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_catSP']
M17_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m17_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k_deact_fwd):
    p, s, cat, cats, inactive_catsp = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat - k_deact_fwd * p * cats
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat - k_deact_fwd * p * cats
    dinactive_catsp_dt = k_deact_fwd * p * cats
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_catsp_dt])

# --- M18 ---
M18_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_cat2S2']
M18_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m18_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k_deact_fwd):
    p, s, cat, cats, inactive_cat2s2 = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat - 2 * k_deact_fwd * cats**2
    dinactive_cat2s2_dt = k_deact_fwd * cats**2
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cat2s2_dt])

# --- M19 ---
M19_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_cat2S']
M19_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1']
@jit(nopython=True)
def hardcoded_m19_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k3):
    p, s, cat, cats, inactive_cat2s = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat - k3 * cat * cats
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat - k3 * cat * cats
    dinactive_cat2s_dt = k3 * cat * cats
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cat2s_dt])

# --- M20 ---
M20_SPECIES_ORDER = ['P', 'S', 'cat', 'catS', 'inactive_cat', 'inactive_catS']
M20_PARAM_NAMES = ['(R1).k1', '(R2).k1', '(R3).k1', '(R4).k1', '(R5).k1', '(R6).k1']
@jit(nopython=True)
def hardcoded_m20_ode(t, concentrations, k1, k_minus_1, k2, k_minus_2, k3, k4):
    p, s, cat, cats, inactive_cat, inactive_cats = concentrations
    dp_dt = k2 * cats - k_minus_2 * p * cat
    ds_dt = -k1 * s * cat + k_minus_1 * cats
    dcat_dt = -k1 * s * cat + k_minus_1 * cats + k2 * cats - k_minus_2 * p * cat - k3 * cat
    dcats_dt = k1 * s * cat - k_minus_1 * cats - k2 * cats + k_minus_2 * p * cat - k4 * cats
    dinactive_cat_dt = k3 * cat
    dinactive_cats_dt = k4 * cats
    return np.array([dp_dt, ds_dt, dcat_dt, dcats_dt, dinactive_cat_dt, dinactive_cats_dt])


# --- Model Information Dictionary ---
MODEL_INFO = {
    "M1": {"ode_func": hardcoded_m1_ode, "species_order": M1_SPECIES_ORDER, "param_names": M1_PARAM_NAMES},
    "M2": {"ode_func": hardcoded_m2_ode, "species_order": M2_SPECIES_ORDER, "param_names": M2_PARAM_NAMES},
    "M3": {"ode_func": hardcoded_m3_ode, "species_order": M3_SPECIES_ORDER, "param_names": M3_PARAM_NAMES},
    "M4": {"ode_func": hardcoded_m4_ode, "species_order": M4_SPECIES_ORDER, "param_names": M4_PARAM_NAMES},
    "M5": {"ode_func": hardcoded_m5_ode, "species_order": M5_SPECIES_ORDER, "param_names": M5_PARAM_NAMES},
    "M6": {"ode_func": hardcoded_m6_ode, "species_order": M6_SPECIES_ORDER, "param_names": M6_PARAM_NAMES},
    "M7": {"ode_func": hardcoded_m7_ode, "species_order": M7_SPECIES_ORDER, "param_names": M7_PARAM_NAMES},
    "M8": {"ode_func": hardcoded_m8_ode, "species_order": M8_SPECIES_ORDER, "param_names": M8_PARAM_NAMES},
    "M9": {"ode_func": hardcoded_m9_ode, "species_order": M9_SPECIES_ORDER, "param_names": M9_PARAM_NAMES},
    "M10": {"ode_func": hardcoded_m10_ode, "species_order": M10_SPECIES_ORDER, "param_names": M10_PARAM_NAMES},
    "M11": {"ode_func": hardcoded_m11_ode, "species_order": M11_SPECIES_ORDER, "param_names": M11_PARAM_NAMES},
    "M12": {"ode_func": hardcoded_m12_ode, "species_order": M12_SPECIES_ORDER, "param_names": M12_PARAM_NAMES},
    "M13": {"ode_func": hardcoded_m13_ode, "species_order": M13_SPECIES_ORDER, "param_names": M13_PARAM_NAMES},
    "M14": {"ode_func": hardcoded_m14_ode, "species_order": M14_SPECIES_ORDER, "param_names": M14_PARAM_NAMES},
    "M15": {"ode_func": hardcoded_m15_ode, "species_order": M15_SPECIES_ORDER, "param_names": M15_PARAM_NAMES},
    "M16": {"ode_func": hardcoded_m16_ode, "species_order": M16_SPECIES_ORDER, "param_names": M16_PARAM_NAMES},
    "M17": {"ode_func": hardcoded_m17_ode, "species_order": M17_SPECIES_ORDER, "param_names": M17_PARAM_NAMES},
    "M18": {"ode_func": hardcoded_m18_ode, "species_order": M18_SPECIES_ORDER, "param_names": M18_PARAM_NAMES},
    "M19": {"ode_func": hardcoded_m19_ode, "species_order": M19_SPECIES_ORDER, "param_names": M19_PARAM_NAMES},
    "M20": {"ode_func": hardcoded_m20_ode, "species_order": M20_SPECIES_ORDER, "param_names": M20_PARAM_NAMES},
}
