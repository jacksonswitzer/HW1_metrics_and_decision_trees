import importlib

def load_student():
    return importlib.import_module('main_roc_student')

def test_roc_auc_public_bonus():
    mod = load_student()
    pairs = [
        (0.95,1),(0.90,1),(0.85,1),(0.80,0),(0.70,1),
        (0.60,0),(0.55,0),(0.50,1),(0.40,0),(0.30,0)
    ]
    try:
        curve = mod.roc_curve(pairs)
        auc_t = round(mod.auc_from_curve(curve), 6)
        auc_r = round(mod.auc_from_ranks(pairs), 6)
    except NotImplementedError:
        return
    assert len(curve) >= 2
    assert (auc_t, auc_r) == (0.840000, 0.840000)
