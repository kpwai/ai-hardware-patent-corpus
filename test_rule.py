#!/usr/bin/env python3
"""
Regression test for the AI hardware corpus rule.

Fixtures are real records from the app folder, hand-labelled. Run this after
ANY change to the vocabularies or CPC prefixes in config.py. A rule that cannot
classify five known cases will not classify 1.5M unknown ones.

    python test_rule.py
"""

import sys

import config as C

# (id, expected_keep, title, abstract, cpc_string)
FIXTURES = [
    ("Fujitsu accelerator/matmul", True,
     "COMPUTING SYSTEM AND METHOD FOR CONTROLLING COMPUTING SYSTEM",
     "A computing system includes: a plurality of accelerators that perform "
     "matrix multiplication computations; a cache memory that caches data of an "
     "external memory that saves a computation result by each of the plurality "
     "of accelerators; and a controller configured to determine whether or not "
     "an access to the cache memory is congested; and control the cache memory "
     "to perform conversion for reducing accuracy of a value of each component "
     "of a matrix on the matrix read from the external memory in response to an "
     "access from one accelerator configured to perform the matrix "
     "multiplication computation.",
     "G06F17/16;;G06F17/16"),

    ("Samsung SDS generative AI messaging", False,
     "METHOD FOR GENERATING RESPONSE MESSAGE USING GENERATIVE AI MODEL",
     "There is provided a method for generating a response message using a "
     "generative AI model, performed by a computing system. The method may "
     "include acquiring a first message with a first user as a recipient, "
     "automatically generating a first prompt for identifying an intention of "
     "the first message, and transmitting the first prompt to a generative "
     "artificial intelligence-based query service.",
     "G06F40/30;;G06N3/0475;;G06N3/006;;G06N5/041"),

    ("Zoom topic relevance", False,
     "Topic Relevance Detection",
     "A conference system automatically detects a topic in a discussion between "
     "two or more participants in a conference based on a transcription of an "
     "audio component of the conference. The conference system determines that "
     "the discussion is a side conversation and schedules a future conference.",
     "G10L15/26;;H04L12/1831;;G06F40/30;;G06N20/00"),

    ("Nanhu confidential computing", False,
     "NOVEL METHOD OF MEASURING CONFIDENTIAL COMPUTING APPLICATION LAYER",
     "Provided is a novel method of measuring a confidential computing "
     "application layer. The method includes utilizing characteristic of static "
     "measurement of an existing underlying component of confidential computing "
     "at a virtual machine level to realize the trusted measurement of a trigger "
     "module TG_APP, and utilizing the trusted TG_APP which is measured by a "
     "confidential computing chip layer.",
     "G06F9/45558;;G06F21/53;;G06F21/57"),

    ("Capital One item recommendations", False,
     "ITEM RECOMMENDATIONS WEIGHTED BY USER-VALUED FEATURES",
     "Computer-implemented methods for determining, sorting, or determining and "
     "sorting a set of items. The method includes receiving, via a user device, "
     "parameter values from a user and receiving parameter weights, and "
     "determining a sort score for each vehicle of a set of vehicles.",
     "G06Q30/0631;;G06Q30/0627;;G06Q30/0629"),

    ("Turner Broadcasting inventory optimisation", False,
     "Managing allocation of inventory mix utilizing an optimization framework",
     "A system is provided that determines reserve inventory units required for "
     "each promotional campaign. Incremental value of revenue from each "
     "inventory utilization type is optimized and ratings for previously "
     "allocated inventory units assigned to a promotion inventory utilization "
     "type is increased. Based on remaining inventory units, schedule of a "
     "channel is communicated to a user device, via a network.",
     "G06Q30/0273;;G06Q30/0254;;G06Q30/0202;;G06Q30/02"),

    ("Deere bale identification (vision, not hardware)", False,
     "BALE IDENTIFICATION USING NET WRAP CHARACTERISTIC",
     "A bale identification system includes an initial image sensor that is "
     "operable to capture an initial image of an initial bale bound with a wrap "
     "material. A computing device analyzes the initial image to identify a "
     "unique wrap characteristic on the wrap material of the initial bale. The "
     "unique wrap characteristic may include a random two-dimensional splotch "
     "pattern formed onto a surface of the wrap material.",
     "G06V10/764;;G06V10/443;;G06V20/64;;G06V20/80;;A01F15/0715"),

    ("Konica Minolta image inspection ('hardware processor')", False,
     "IMAGE INSPECTION APPARATUS, IMAGE INSPECTION METHOD",
     "This image inspection apparatus includes a hardware processor that "
     "generates a reference image to be used for inspection of an image formed "
     "on a recording medium; executes inspection of the image formed on the "
     "recording medium based on the reference image; and receives an execution "
     "instruction to execute inspection before generation of the reference "
     "image is completed.",
     "G06T2207/30144;;G06T11/00;;G06T7/001"),

    ("Quantum bit array (G06N10 quantum, not AI)", False,
     "QUANTUM BIT ARRAY AND QUANTUM COMPUTER",
     "A quantum bit array includes a plurality of qubits arranged on a "
     "semiconductor substrate, and a control circuit configured to apply "
     "control pulses to the qubits. A neural network may be used to calibrate "
     "the control pulses applied to the quantum computer.",
     "G06N10/40;;G06N10/20;;H01L29/66"),

    ("Mask layout via ML (AI for hardware, not hardware for AI)", False,
     "MASK LAYOUT DETERMINING MODEL TRAINING METHOD AND APPARATUS",
     "A mask layout determining model training method includes obtaining "
     "training data comprising semiconductor wafer patterns, training a neural "
     "network to predict an optical proximity correction for a mask layout, "
     "and determining a mask layout for a photolithography process.",
     "G06N3/08;;H01L21/027;;G03F1/36"),

    ("Memristive synapse describing its own fabrication (must NOT be vetoed)", True,
     "MEMRISTIVE SYNAPTIC DEVICE FOR NEUROMORPHIC COMPUTING",
     "A memristive synaptic device includes a crossbar array of resistive "
     "switching elements formed on a semiconductor substrate. The device is "
     "fabricated by photolithography and etching processes to define the "
     "electrode pattern. The synaptic weight is encoded in the conductance of "
     "each element, enabling in-memory computing for a spiking neural network.",
     "G06N3/063;;H10N70/20"),
]


def any_prefix(codes, prefixes):
    return any(c.startswith(p) for c in codes for p in prefixes)


def classify(title, abstract, cpc):
    codes = sorted({c.strip() for c in cpc.split(";;") if c.strip()})
    text = f"{title}. {abstract}"

    t1_pref = tuple(C.TIER1_CPC_PREFIXES)
    if C.TIER1_INCLUDE_ARCHITECTURE:
        t1_pref += tuple(C.TIER1_ARCHITECTURE_PREFIXES)

    ev = {
        "tier1": any_prefix(codes, t1_pref),
        "ai_cpc": any_prefix(codes, C.AI_CPC_PREFIXES),
        "hw_cpc": any_prefix(codes, C.HW_CPC_PREFIXES),
        "ai_text": bool(C.AI_TEXT_RE.search(text)),
        "hw_text": bool(C.HW_TEXT_RE.search(text)),
    }
    ev["ai_mech"] = bool(getattr(C, "AI_MECHANISM_RE", None)
                         and C.AI_MECHANISM_RE.search(text))
    veto_target = title if getattr(C, "VETO_SCOPE", "title") == "title" else text
    ev["veto"] = bool(getattr(C, "APPLY_SUBJECT_VETO", False)
                      and C.EXCLUDE_SUBJECT_RE.search(veto_target))
    if any_prefix(codes, getattr(C, "AI_CPC_EXCLUDE", ())):
        ev["ai_cpc"] = False

    cond_ai = ev["ai_cpc"] or ev["ai_text"]
    if getattr(C, "AI_MECHANISM_COUNTS_AS_AI", False):
        cond_ai = cond_ai or ev["ai_mech"]
    cond_hw = ev["hw_cpc"] or ev["hw_text"]
    keep = ev["tier1"] or (cond_ai and cond_hw)
    if ev["veto"]:
        keep = False
    return keep, cond_ai, cond_hw, ev


def main():
    print("=" * 78)
    print("CORPUS RULE REGRESSION TEST")
    print(f"  AI_MECHANISM_COUNTS_AS_AI = "
          f"{getattr(C, 'AI_MECHANISM_COUNTS_AS_AI', 'not defined')}")
    print("=" * 78)

    failures = 0
    for name, expected, title, abstract, cpc in FIXTURES:
        keep, cond_ai, cond_hw, ev = classify(title, abstract, cpc)
        ok = keep == expected
        failures += not ok
        mark = "PASS" if ok else "FAIL"
        print(f"\n  [{mark}] {name}")
        print(f"        expected keep={expected}  got keep={keep}")
        print(f"        AI  cpc={ev['ai_cpc']!s:5} text={ev['ai_text']!s:5} "
              f"mech={ev['ai_mech']!s:5} -> cond_ai={cond_ai}")
        print(f"        HW  cpc={ev['hw_cpc']!s:5} text={ev['hw_text']!s:5} "
              f"            -> cond_hw={cond_hw}")
        if not ok:
            if expected and not cond_ai:
                print("        >> FALSE NEGATIVE: no AI signal detected. The patent")
                print("           describes the mechanism without naming AI.")
            elif expected and not cond_hw:
                print("        >> FALSE NEGATIVE: no hardware signal detected.")
            elif not expected:
                print("        >> FALSE POSITIVE: rule is too permissive here.")

    print("\n" + "=" * 78)
    print(f"  {len(FIXTURES) - failures}/{len(FIXTURES)} passed")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())