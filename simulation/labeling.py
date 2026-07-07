"""Map truth-based blendedness metric to the 4 ordinal blend-severity classes."""

# PLACEHOLDER thresholds -- tune these against the calibration thumbnails
# (calibration_thumbnails.py) before generating the full training set.
DEFAULT_THRESHOLDS = {
    "clean_max": 0.03,      # B < clean_max            -> class 1 (clean/isolated)
    "moderate_max": 0.15,   # clean_max <= B < this     -> class 2 (moderate, disentangleable)
    "severe_max": 0.5,      # moderate_max <= B < this  -> class 3 (severe)
    # B >= severe_max                                   -> class 4 (ambiguous)
}

CLASS_NAMES = {
    1: "clean/isolated",
    2: "moderate blend",
    3: "severe blend",
    4: "ambiguous overlap",
}


def classify(blendedness, thresholds=None):
    t = thresholds or DEFAULT_THRESHOLDS
    if blendedness < t["clean_max"]:
        return 1
    elif blendedness < t["moderate_max"]:
        return 2
    elif blendedness < t["severe_max"]:
        return 3
    else:
        return 4
