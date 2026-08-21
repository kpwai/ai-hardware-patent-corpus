"""
Single source of truth for the AI hardware patent pipeline.

Every hyperparameter, path, and vocabulary lives here. No stage script hardcodes
anything. This is what makes the reproducibility statement honest: the config
is dumped into the run manifest at every stage.
"""

from pathlib import Path
import re

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT = Path("PATH/TO/OUTPUT")   # edit this

# The raw export is two products, each split into per-year CSVs.
RAW_APP_DIR = Path("PATH/TO/LENS/filing")   # edit this
RAW_GRANT_DIR = Path("PATH/TO/LENS/grant")     # edit this

# How to combine the two products.
#   "invention"    -- one record per application number, grant metadata
#                     preferred when both exist, is_granted flag retained.
#                     RECOMMENDED: no double counting, full temporal coverage,
#                     and grant status becomes a free covariate.
#   "grants_only"  -- quality-filtered but lags 2-4 years and carries
#                     grant-rate bias across firms and jurisdictions
#   "apps_only"    -- best temporal coverage for early detection, but
#                     includes abandoned filings
CORPUS_MODE = "invention"

# Application numbers appear as e.g. "US 33444406 A" -- normalise before joining.
APPNO_CLEAN_RE = r"[\s\-/,\.]"

# Deduplication level.
#   "publication" -- collapse exact Lens ID duplicates only
#   "application" -- one record per application number (per-office invention)
#   "family"      -- one record per earliest priority number. REQUIRED if the
#                    export spans multiple jurisdictions, since the same
#                    invention filed in US/EP/CN/JP/KR carries five different
#                    application numbers but one priority. This is what the
#                    reviewer meant by "family-level normalization".
DEDUP_LEVEL = "family"

# Restrict to a single office. Stage 0 found US 93%, EP 5%, DE 2%, CN ~0%.
# CN is negligible, so the post-2015 Chinese subsidy confound does not apply --
# but EP/DE records are largely family members of the same US inventions, and
# restricting to one office gives one examination standard and one filing-
# incentive regime for 93% of the data. Set to None to keep all offices.
JURISDICTION_FILTER = "US"

# Document types present in the export, from Stage 0:
#   Granted Patent, Patent Application, Search Report, Statutory Invention
#   Registration, Amended Application, Limited Patent, Amended Patent,
#   Abstract, Unknown, Ambiguous
# A search report is not an invention; neither is an abstract-only publication.
# Keep only document types that represent a patent or patent application.
DOC_TYPES_KEEP = {
    "granted patent",
    "patent application",
    "amended patent",
    "amended application",
    "limited patent",
    "statutory invention registration",
}
DOC_TYPES_DROP = {
    "search report",       # EP/PCT search reports -- not inventions
    "abstract",            # abstract-only publications
    "unknown",
    "ambiguous",
}

# The sweep also included G06Q (administrative, commercial, financial and
# managerial data processing). That subclass carries essentially no AI hardware
# and simply inflates the raw denominator; the two-condition rule removes it.
# Recorded here so the paper can state the sweep honestly.
SWEEP_SUBCLASSES = ("G06F", "G06T", "G06V", "G06K", "G06N", "G06Q")

STAGE1 = ROOT / "stage1_corpus"
STAGE2 = ROOT / "stage2_embeddings"
STAGE3 = ROOT / "stage3_clusters"
STAGE4 = ROOT / "stage4_analysis"
FIGURES = ROOT / "figures"

for _p in (ROOT, STAGE1, STAGE2, STAGE3, STAGE4, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# The join key used everywhere. Never align by row position.
ID_COL = "Lens ID"

# ---------------------------------------------------------------------------
# STAGE 1 -- CORPUS CONSTRUCTION
# ---------------------------------------------------------------------------
# The raw export was filtered on CPC subclasses G06F/G06T/G06V/G06K/G06N, which
# selects *computing*, not AI hardware. G06F alone is all of electric digital
# data processing. We recover an AI hardware corpus with a two-condition rule.

# Tier 1: unconditional. G06N3/06 covers the physical realisation of neural
# networks -- these are AI hardware by the definition of the code itself.
TIER1_CPC_PREFIXES = ("G06N3/06",)

# Optional widening of Tier 1. G06N3/04 is network architecture, which is often
# but not always hardware. Off by default; enable to test sensitivity.
TIER1_INCLUDE_ARCHITECTURE = False
TIER1_ARCHITECTURE_PREFIXES = ("G06N3/04",)

# ---------------------------------------------------------------------------
# THE RULE (revised after human validation of 50 records)
# ---------------------------------------------------------------------------
# Measured precision of the candidate rules against human labels:
#
#   AI-cond & HW-cond, either satisfiable by CPC   42.4%  (the original design)
#   T2 requiring both texts                        61.9%
#   G06N3/06 alone                                 68.4%
#   G06N3/06 AND text corroboration                80.0%  <- adopted
#   G06N3/06 AND hardware text                    100.0%  but recall 40%
#
# Two findings drove this. Tier 2 contributed ZERO true positives in the
# labelled sample -- every real AI hardware patent already carried G06N3/06 --
# while adding 19 false positives. And every false positive met both conditions
# on CPC CODES ALONE, with no textual evidence either way: G06N subgroups other
# than /06 (learning methods, ML generally, knowledge-based systems) are
# software, and hardware codes like G06F9/38 get assigned for incidental
# implementation detail.
#
# The rule is therefore: the definitional hardware code, corroborated by text.
ENABLE_TIER2 = False
TIER1_REQUIRE_TEXT_CORROBORATION = True

# Condition A evidence: AI-bearing CPC
AI_CPC_PREFIXES = ("G06N",)

# ...but G06N10/00 is QUANTUM COMPUTING, not AI. It sits inside the G06N
# prefix and admitted quantum bit arrays, quantum trend detection, and
# cryogenic attenuators. Quantum is a distinct technology domain and is absent
# from the architecture taxonomy, so those records would be unlabelled anyway.
AI_CPC_EXCLUDE = ("G06N10",)

# ---------------------------------------------------------------------------
# SUBJECT-MATTER VETO
# ---------------------------------------------------------------------------
# Two classes of patent satisfy both conditions without being AI hardware.
#
# QUANTUM: quantum computing hardware. Out of scope, as above.
#
# EDA / FAB: machine learning applied TO chip design and manufacturing --
# mask layout optimisation, lithography, wafer inspection, yield prediction,
# place-and-route. These are "AI for hardware", the inverse of this study's
# subject, and they match because `wafer` and `semiconductor` are strong
# hardware terms while the abstract discusses model training. Distinguishing
# the two directions is a substantive scope decision; state it in the paper.
EXCLUDE_SUBJECT_RE = re.compile(
    r"\b(?:"
    r"qubit|quantum\s+(?:bit|computer|computing|circuit|gate|processor|"
    r"annealing|entanglement|superposition)|"
    r"superconducting\s+(?:qubit|resonator)|cryogenic\s+(?:attenuat|amplif)|"
    r"mask\s+(?:layout|pattern|synthes\w*)|"
    r"optical\s+proximity\s+correction|"
    # Bare "lithograph*" vetoes titles like "Lithographic memristive array",
    # where the word is a fabrication adjective rather than the subject.
    # Only computational-lithography phrasings indicate an EDA patent.
    r"inverse\s+lithograph\w*|computational\s+lithograph\w*|"
    r"lithograph\w*\s+(?:apparatus|simulation|model)|"
    r"place[\s\-]and[\s\-]route|"
    r"netlist|design\s+rule\s+check|"
    r"electronic\s+design\s+automation|"
    r"wafer\s+(?:inspection|defect|yield|map)|"
    r"etching\s+process"
    r")\b",
    re.IGNORECASE,
)
APPLY_SUBJECT_VETO = True

# Where the veto is evaluated.
#   "title"  -- RECOMMENDED. A patent whose SUBJECT is quantum computing or
#               mask-layout optimisation says so in its title. Scoping to the
#               title avoids vetoing genuine AI hardware on incidental
#               mentions: memristor and neuromorphic device patents routinely
#               describe their own photolithographic fabrication, and an
#               abstract-scoped veto removed 39 Tier 1 (G06N3/06) records.
#   "text"   -- title + abstract. Higher recall on true EDA/quantum, but
#               vetoes real device patents.
VETO_SCOPE = "title"

# Condition B evidence: hardware-bearing CPC. Broad on purpose -- the text
# condition does the discriminating.
HW_CPC_PREFIXES = (
    # Bare G06F9/ ("arrangements for program control") covers virtual machines,
    # resource allocation and task scheduling -- all software -- and matched
    # 372,565 inventions on its own. Only the instruction-set and execution
    # groups under it denote machine structure.
    "G06F9/30",  # instruction set architecture
    "G06F9/38",  # instruction execution, pipelining, superscalar
    "G06F7/",    # arithmetic and logic units, multipliers
    "G06F1/",    # power, clocking, thermal
    "G06F13/",   # bus, interconnect, DMA
    # Bare G06F15/ includes G06F15/16 (networked computers); only /76 onward
    # denotes machine architecture.
    "G06F15/76", # architectures with functional units
    "G06F15/78", # single-chip architectures
    "G06F15/80", # SIMD / array processors
    "G06F15/82", # dataflow architectures
    "G06F12/",   # memory addressing and cache structures
    "G11C",      # memory: SRAM, DRAM, ReRAM, MRAM, flash
    "H10N",      # post-2023 successor class for memristive / neuromorphic
    "H01L",      # semiconductor devices (pre-restructuring)
    "H03K",      # pulse technique
    "H03M",      # coding, decoding
    "G06E",      # optical computing
    "G06G",      # analogue computers
)

AI_TEXT_RE = re.compile(
    r"\b(?:"
    r"neural\s+network(?:s)?|deep\s+neural|deep\s+learning|machine\s+learning|"
    r"convolutional\s+(?:neural\s+)?network|recurrent\s+neural|long\s+short[\s\-]term\s+memory|"
    r"transformer\s+(?:model|architecture|network)|attention\s+mechanism|self[\s\-]attention|"
    r"artificial\s+intelligence|neuromorphic|spiking\s+neural|synaptic|"
    r"activation\s+function|backpropagation|back[\s\-]propagation|"
    r"inference\s+(?:engine|accelerator|operation|phase)|model\s+training|"
    r"reinforcement\s+learning|federated\s+learning|"
    r"weight\s+(?:matrix|matrices|quantization)|feature\s+map"
    r")\b",
    re.IGNORECASE,
)

# Mechanism pathway. Many hardware patents -- especially JP/KR filings --
# describe the computation without ever naming AI. The Fujitsu accelerator
# patent ("a plurality of accelerators that perform matrix multiplication
# computations ... conversion for reducing accuracy") is textbook AI hardware
# and contains none of the vocabulary above. These terms recover that class.
#
# They are deliberately kept SEPARATE from AI_TEXT_RE so the evidence
# composition table can report how many records enter via each pathway. If
# validation shows the mechanism pathway is noisy, set the flag below to False
# and the rule reverts without touching anything else.
AI_MECHANISM_RE = re.compile(
    r"\b(?:"
    r"matrix\s+(?:multiplication|multiply|multiplier)|"
    r"matrix[\s\-]vector\s+(?:multiplication|product)|"
    r"tensor\s+(?:core|unit|operation|comput|processing)|"
    r"systolic|multiply[\s\-]accumulate|mac\s+(?:unit|array)|"
    r"gemm|dot[\s\-]product\s+engine|"
    r"weight[\s\-]stationary|output[\s\-]stationary|dataflow\s+architecture|"
    # Precision terms must be tied to weights/activations/models, otherwise
    # they match audio and video codec quantization -- "vector quantization
    # using thresholds", "digital signal encoding with quantizing" etc.
    r"(?:weight|activation|model|network|post[\s\-]training)[\s\-]"
    r"(?:quantiz\w+|precision)|"
    r"quantiz\w+\s+(?:neural|network|model|weight)|"
    r"(?:reduced|low|mixed)[\s\-]precision\s+"
    r"(?:arithmetic|comput|inference|training|multiplier|unit)|"
    r"bfloat|fp16|int8|"
    r"convolution\s+(?:engine|accelerat|unit)"
    r")\b",
    re.IGNORECASE,
)

# Whether the mechanism pathway alone satisfies the AI condition.
# True  -> recovers mechanism-described hardware (better recall)
# False -> requires explicit AI vocabulary or a G06N code (better precision)
AI_MECHANISM_COUNTS_AS_AI = True

# Hardware vocabulary is split by discriminative power.
#
# WEAK terms are boilerplate. US software claims routinely recite "a processor
# and a memory storing instructions", so `processor`, `circuit`, `memory`,
# `cache`, `chip` and `hardware` are satisfied by essentially every software
# patent. Using them as evidence matched 25.5% of the whole sweep and drove
# retained precision down to roughly 12%.
#
# STRONG terms denote a physical compute structure that is the subject of the
# invention rather than the platform it runs on. Only these satisfy the
# hardware condition. Weak terms are still flagged, for reporting only.
HW_TEXT_STRONG_RE = re.compile(
    r"\b(?:"
    # specialised compute structures
    r"accelerator|systolic\s+array|processing\s+element\s+array|pe\s+array|"
    r"tensor\s+core|mac\s+(?:unit|array)|multiply[\s\-]accumulate\s+unit|"
    r"arithmetic\s+logic\s+unit|dataflow\s+architecture|"
    r"neural\s+(?:processor|processing\s+unit|network\s+chip|network\s+circuit)|"
    r"hardware\s+(?:accelerat\w+|implementation\s+of\s+a\s+neural)|"
    # device physics / emerging memory
    r"memristor|memristive|crossbar|neuromorphic|spiking\s+circuit|"
    r"synaptic\s+(?:device|circuit|element|array|weight)|"
    r"phase[\s\-]change\s+memory|resistive\s+ram|reram|rram|mram|"
    # memory-centric compute
    r"in[\s\-]memory\s+comput\w+|processing[\s\-]in[\s\-]memory|"
    r"compute[\s\-]in[\s\-]memory|near[\s\-]memory\s+(?:comput|processing)|"
    r"near[\s\-]data\s+processing|high[\s\-]bandwidth\s+memory|"
    r"memory\s+(?:array|bank|die)|sram\s+array|dram\s+array|"
    # named silicon
    r"asic|fpga|cpld|gpgpu|system[\s\-]on[\s\-]chip|chiplet|"
    r"application[\s\-]specific\s+integrated\s+circuit|"
    r"field[\s\-]programmable\s+gate\s+array|"
    r"graphics\s+processing\s+unit|tensor\s+processing\s+unit|"
    r"digital\s+signal\s+processor|vision\s+processing\s+unit|"
    # fabrication / device level
    r"semiconductor|wafer|transistor|integrated\s+circuit|"
    r"silicon\s+(?:die|substrate|area)|on[\s\-]chip\s+(?:memory|network|buffer)|"
    # alternative computing substrates
    r"analog\s+comput\w+|photonic\s+comput\w+|optical\s+(?:neural|comput)"
    r")\b",
    re.IGNORECASE,
)

# Retained for diagnostics only -- NOT used to satisfy the hardware condition.
HW_TEXT_WEAK_RE = re.compile(
    r"\b(?:processor|circuit(?:ry)?|memory|cache|chip|hardware|"
    r"computing\s+device|register\s+file|pipeline\s+stage|interconnect)\b",
    re.IGNORECASE,
)

# Backwards-compatible alias used by test_rule.py
HW_TEXT_RE = HW_TEXT_STRONG_RE

# Deduplication: the export mixes A1 applications and B1/B2 grants of the same
# invention, which double-counts and spreads the duplicates across years.
DEDUP_KEY = "Application Number"        # or "Earliest Priority Date"
DEDUP_PREFER_GRANTED = True

# ---------------------------------------------------------------------------
# STAGE 2 -- EMBEDDINGS
# ---------------------------------------------------------------------------
# PatentSBERTa is domain-adapted and defensible in review. all-mpnet-base-v2 is
# the general-purpose fallback. Both are 768-d.
EMBED_MODEL = "AI-Growth-Lab/PatentSBERTa"
EMBED_FALLBACK = "sentence-transformers/all-mpnet-base-v2"
EMBED_BATCH = 256
EMBED_MAX_SEQ = 256          # title + abstract rarely needs more
EMBED_FP16 = True
EMBED_NORMALIZE = True       # unit-norm at source; view weights then mean what they say
EMBED_CHECKPOINT_EVERY = 50_000

# ---------------------------------------------------------------------------
# STAGE 3 -- MULTI-VIEW CLUSTERING
# ---------------------------------------------------------------------------
# Each view is L2-normalised, then scaled. Because every block is unit-norm,
# these weights ARE the relative contribution to the cosine numerator.
#
# Chosen from the Stage 3 ablation (stage3_clusters/ablation.csv):
#
#   TEMPORAL = 0. eta^2(year) is flat at ~0.10 across temporal weights 0 to
#   0.60, so injecting time does NOT make clusters more time-aligned -- the
#   temporal structure is intrinsic to the technologies. But ARI drops to ~0.64
#   as soon as any temporal weight is applied and stays there, so injection
#   perturbs the partition without improving it. Excluding it is the defensible
#   choice; time remains an analysis dimension in Stage 4.
#
#   CPC = 0.30. Load-bearing, unlike the previous pipeline where it was ~2% of
#   the signal and inert. At weight 0 the clustering degenerates to 3 clusters
#   with 1.6% noise; at 0.15-0.30 it yields 20-22 clusters with silhouette
#   ~0.47-0.50. Non-monotonic: 0.60 collapses again to 4 clusters.
#
# Note for the paper: eta^2(year) rises from 0.012 (no CPC) to ~0.10 (with
# CPC), so the temporal alignment of clusters comes from the classification
# scheme, not the time view -- CPC codes partially encode era because the
# scheme itself evolves.
# Keep cpc at 0.30. At weight 1.0 the CPC block dominates and HDBSCAN simply
# rediscovers the classification taxonomy: silhouette 0.83 with 0.8% noise is
# the signature of clustering on a categorical variable, not of finding
# semantic structure.
VIEW_WEIGHTS = {"semantic": 1.00, "cpc": 0.30, "temporal": 0.00}

# CPC view granularity: 4 = subclass (G06N), 8 = main group (G06N3/063).
#
# Set to 8. At subclass level the corpus has only 160 distinct codes at 2.33
# per patent -- almost everything is G06N, so the structural view barely
# discriminates and UMAP produces one blob. Main-group level gives 1,788 codes
# and separates G06N3/063 from G06N3/0464 from G11C11/54, which is what
# recovers a usable partition.
CPC_GRANULARITY = 8

# Which year field drives the temporal view and all longitudinal analysis.
# Filing year leads publication year by ~2-4 years; for early-detection claims
# filing year is the correct basis. Publication year is the robustness check.
YEAR_COL_PRIMARY = "filing_year"
YEAR_COL_ROBUSTNESS = "pub_year"

# n_neighbors=10 preserves local structure, which a homogeneous 7,464-record
# corpus needs; 30 emphasises global structure and merged the clusters.
UMAP_PARAMS = dict(n_neighbors=10, min_dist=0.0, n_components=10,
                   metric="cosine", low_memory=True)
UMAP_VIZ_PARAMS = dict(n_neighbors=30, min_dist=0.1, n_components=2,
                       metric="cosine", low_memory=True)
# NOTE: these must be tuned against the SAME view weights used for the final
# run. A sweep run on an embedding that included the temporal view does not
# transfer once temporal is set to 0: with min_samples=5 and no temporal
# component the density estimate becomes permissive enough that excess-of-mass
# selects a single giant cluster (2 clusters, 0.4% noise, silhouette 0.446 --
# degenerate). min_samples=10 restores structure.
#
# Validated by the ablation at temporal=0, cpc=0.30: 20 clusters, 41.4% noise,
# silhouette 0.495.
# From rep_sweep.py at gran=8, w_cpc=0.30, nn=10:
#   40 clusters, 36.0% noise, silhouette 0.511, largest cluster 4.5%
HDBSCAN_PARAMS = dict(min_cluster_size=50, min_samples=5)

# Ablation grid for the temporal-injection question the reviewer raised.
ABLATION_TIME_WEIGHTS = [0.0, 0.05, 0.15, 0.30, 0.60]
ABLATION_CPC_WEIGHTS = [0.0, 0.15, 0.30, 0.60]

SEED = 42

# ---------------------------------------------------------------------------
# STAGE 4 -- ARCHITECTURE TAXONOMY
# ---------------------------------------------------------------------------
# CRITICAL: abbreviations are matched with WORD BOUNDARIES. Plain substring
# matching silently matches 'tpu' inside "output", 'npu' inside "input", and
# 'asic' inside "basic", which destroyed two of six series in the v1 analysis.
ARCH_TAXONOMY = {
    "CPU": {
        "abbrev": ["cpu"],
        "phrase": ["central processing unit", "general purpose processor",
                   "instruction set architecture", "scalar processor"],
    },
    "GPU": {
        "abbrev": ["gpu", "gpgpu", "cuda"],
        "phrase": ["graphics processing unit", "graphics processor",
                   "shader core", "streaming multiprocessor"],
    },
    "TPU/NPU": {
        "abbrev": ["tpu", "npu"],
        # Terms below marked (+) were added after validation showed the keyword
        # matcher recovered only 29% of human-identified TPU/NPU patents.
        "phrase": ["tensor processing unit", "neural processing unit",
                   "neural processor", "systolic array",
                   "neural network accelerator", "deep learning accelerator",
                   "inference engine", "inference accelerator",
                   "ai accelerator", "artificial intelligence accelerator",
                   "neural network chip", "ai chip",
                   "processing element array", "mac array",
                   "multiply accumulate array", "tensor core",
                   "convolution engine", "convolution accelerator",
                   "matrix multiplication unit", "dataflow accelerator",
                   # (+) stem forms: "convolution acceleration" was missed
                   "convolution accelerat", "convolution operation",
                   "convolution comput",
                   # (+) "neural processing engine", "neural engine"
                   "neural processing engine", "neural engine",
                   "neural network processor", "neural network engine",
                   # (+) bare multiply-accumulate, seen as "operation device",
                   #     "operation circuit", "operation system"
                   "multiply accumulate", "multiply and accumulate",
                   "arithmetic logic unit", "arithmetic circuit",
                   "arithmetic operation circuit",
                   # (+) hardware-structure framing common in JP/KR filings
                   "hardware structure of", "hardware accelerator",
                   "fixed point scale", "fixed point implementation"],
    },
    "FPGA/ASIC": {
        "abbrev": ["fpga", "asic", "cpld"],
        "phrase": ["field programmable gate array",
                   "application specific integrated circuit",
                   "reconfigurable logic", "reconfigurable computing",
                   "programmable logic device", "lookup table based"],
    },
    # NEW. G06N3/06 (physical realisation of neural networks) is 11,256 of the
    # 18,307-record corpus and is overwhelmingly device-level neuromorphic and
    # memristive work. Without this class most of Tier 1 was unlabelled.
    "Neuromorphic/Memristive": {
        "abbrev": ["rram", "reram", "pcm"],
        "phrase": ["neuromorphic", "memristor", "memristive",
                   "spiking neural", "spiking neuron", "neuron circuit",
                   "artificial synapse", "synaptic device", "synaptic array",
                   "synaptic weight", "synaptic element", "synapse circuit",
                   "crossbar array", "crossbar structure",
                   "resistive switching", "resistive random access memory",
                   "phase change memory", "ferroelectric synapse",
                   "leaky integrate and fire", "brain inspired",
                   # (+) bare forms: "artificial neuron device", "synapses
                   #     having memory cells", "each neuron"
                   "artificial neuron", "neuron device", "synapse",
                   # (+) IBM's resistive processing unit, missed twice
                   "resistive processing unit"],
    },
    "PIM/Memory-Centric": {
        "abbrev": ["pim", "hbm", "cim"],
        "phrase": ["processing in memory", "process in memory",
                   "compute in memory", "computing in memory",
                   "in memory computing", "in memory computation",
                   "near memory", "near data processing",
                   "high bandwidth memory", "memory centric",
                   "computational memory", "associative memory device",
                   "analog in memory",
                   # (+) "NAND memory arrays" realising a binary neural net.
                   # NOTE: bare "memory array" and "memory bank" were tried and
                   # removed -- they fired on ordinary memory patents and cut
                   # PIM precision to 44% (5 FP of 9) while adding no recall,
                   # which was already 100% without them.
                   "nand memory array", "nand array",
                   "resistive processing unit"],
    },
    "DPU/VPU/DSP": {
        "abbrev": ["dpu", "vpu", "dsp"],
        "phrase": ["data processing unit", "vision processing unit",
                   "digital signal processor", "image signal processor"],
    },
    "Analog/Photonic": {
        "abbrev": [],
        "phrase": ["photonic comput", "optical neural", "optical comput",
                   "analog comput", "analog accelerat", "optical accelerat",
                   "silicon photonic", "mach zehnder", "analog matrix",
                   "charge domain", "current mode comput"],
    },
}


def build_arch_patterns(taxonomy=None, granularity=None):
    """Compile boundary-safe patterns. Abbreviations get \\b guards and an
    optional plural; phrases tolerate hyphen or whitespace between tokens."""
    tax = taxonomy or ARCH_TAXONOMY
    out = {}
    for cat, spec in tax.items():
        parts = []
        for a in spec.get("abbrev", []):
            parts.append(r"\b" + re.escape(a) + r"s?\b")
        for p in spec.get("phrase", []):
            toks = [re.escape(t) for t in p.split()]
            parts.append(r"\b" + r"[\s\-]+".join(toks) + r"\w*")
        out[cat] = re.compile("|".join(parts), re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# FIRM-LEVEL ANALYSIS -- REMOVED FROM SCOPE
# ---------------------------------------------------------------------------
# RTA and assignee specialisation are out of scope for this submission. Lens
# owner strings are not disambiguated, and doing entity resolution properly
# (variant matching, subsidiary rollup, name changes over time) is a project in
# itself. A reviewer asked for "robust entity resolution, thresholding, and
# uncertainty estimates" -- three demands that cannot be met credibly in the
# time available. State it as a limitation and future work rather than shipping
# a weak version.
#
# The replacement contribution is the naive-vs-validated comparison in Stage 4:
# what changes in the reported trends when the corpus is constructed properly.

# ---------------------------------------------------------------------------
# ANALYSIS WINDOW
# ---------------------------------------------------------------------------
YEAR_START = 2010
# Stage 1 measured publication lag: median 2.0y, mean 2.55y, p90 5.0y, against
# an export running to 2026. A filing year Y is therefore ~90% published only
# by Y+5, i.e. complete through 2021. The apparent peak at filing year 2020 and
# the decline after it are RIGHT TRUNCATION, not a fall in filings.
YEAR_END = 2021
YEAR_END_DISPLAY = 2024     # shown, but shaded as incomplete

VALIDATION_SAMPLE_SIZE = 200

# Minimum validated records in a year for its share to enter the trend
# comparison. After the family-dedup fix the corpus is 7,464, so a threshold of
# 300 would leave only 4 usable years. At 100 the window opens at 2016.
TREND_MIN_N = 100