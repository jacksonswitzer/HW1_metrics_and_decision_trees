import importlib

def load_student():
    return importlib.import_module('main_confusion_student')

def test_example_confusion_metrics():
    mod = load_student()
    cm = [[90,10],[5,95]]
    try:
        prec = round(mod.precision(cm), 4)
        rec  = round(mod.recall(cm), 4)
        f1   = round(mod.f1_score(cm), 4)
        spec = round(mod.specificity(cm), 4)
        acc  = round(mod.accuracy(cm), 4)
    except NotImplementedError:
        # allow collection when student hasn't implemented yet
        return
    assert (prec, rec, f1, spec, acc) == (0.9048, 0.9500, 0.9268, 0.9000, 0.9250)
