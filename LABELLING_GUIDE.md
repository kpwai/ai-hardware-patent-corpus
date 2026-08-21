# Labelling Guide - AI Hardware Validation Sample

200 rows, three columns each. Budget 1-2 minutes per row: **3-6 hours total.**
Read the title and abstract; do not open the patent. If a row takes more than
two minutes, mark `BORDERLINE` and move on - agonising over hard cases costs
more than it buys, and BORDERLINE is reported separately.

Label the **blind** file (`validation_blind.csv`), not the sample directly. It
hides whether the rule kept each record, which is the quantity you're
estimating.

```bash
python blind_labels.py make      # once
# ... label validation_blind.csv in a spreadsheet ...
python blind_labels.py merge     # when done, or at any checkpoint
python stage4_analysis.py
```

---

## Column 1 - `is_ai_hardware`  (Y / N / BORDERLINE)

**The test:** does the invention claim a *physical computing structure* whose
purpose is to execute or accelerate AI/ML computation?

Two things must both be true. The subject is hardware - a circuit, device,
memory array, chip architecture, or dataflow organisation. And its purpose is
AI - the novelty is tied to neural network, ML, or inference workloads.

### Y - clear cases
- Neural network accelerators, NPUs, TPUs, inference engines
- Neuromorphic circuits, artificial synapses, memristive crossbars, spiking neurons
- Processing-in-memory, compute-in-memory, associative memory devices
- Systolic arrays, MAC arrays, tensor cores, convolution engines
- Analog or photonic matrix-multiplication hardware
- Memory devices whose novelty is serving AI workloads (HBM for training, weight storage)
- **Arithmetic circuits specialised for neural computation** - e.g. a sparse
  matrix multiply unit for convolutional layers

### N - clear cases
- ML algorithms, model architectures, training methods (no hardware novelty)
- ML applied to a domain: medical imaging, recommendations, speech, vehicles
- Software running on generic processors - "a processor and memory storing
  instructions" is boilerplate, not hardware novelty
- **AI *for* hardware**: mask layout, lithography, wafer inspection, EDA, yield
  prediction. The inverse of this study's subject.
- Quantum computing
- Hardware with no AI purpose: port replicators, display drivers, cache
  coherence, storage controllers
- Devices that merely *contain* an AI chip, where the claim is about the
  application (a vehicle, a phone, a camera)

### BORDERLINE - use freely, don't agonise
- Compilers, schedulers, mappers that target accelerators
- Model compression or quantisation *algorithms* with no claimed circuit
- Dataflow scheduling described without a claimed structure
- Generic memory devices where the AI connection is implied but not claimed

**Consistency rules for cases you'll hit repeatedly:**

| Situation | Label |
|---|---|
| Method claim, but the novelty is a described circuit | Y |
| Method claim executed on a generic processor | N |
| Device claim where the AI part is off-the-shelf | N |
| Hardware for AI | Y |
| AI for hardware | N |
| Memory array + explicit compute function | Y |
| Memory array, storage only | N |

---

## Column 2 - `architecture`

One of: `CPU` `GPU` `TPU-NPU` `FPGA-ASIC` `NEUROMORPHIC` `PIM` `DPU-VPU-DSP`
`ANALOG` `NONE`

Multiple allowed, comma-separated (`PIM,NEUROMORPHIC`).

Use `NONE` when the patent is AI hardware but names no architecture family -
a generic accelerator, an unspecified circuit. **This will be common** and
that's expected: ~80% of the corpus carries no keyword label, and this column
measures whether that reflects the patents or a gap in the vocabulary.

If you find yourself wanting a category that doesn't exist, write it in
`notes`. That's evidence the taxonomy needs extending.

---

## Column 3 - `is_about_class`  (Y / N)

Only fill this when `architecture` is not `NONE`.

**Y** - the patent is *about* that architecture. **N** - it merely mentions it,
typically in background or as a comparison ("unlike conventional GPUs...").

This measures the mention-versus-subject problem directly. A reviewer will
raise it if you don't.

---

## Order of work

1. **Do all rows in order.** The blind file is shuffled, so retained and
   excluded rows are interleaved. Don't skip ahead.
2. **Checkpoint every ~50 rows**: run `merge`, then `stage4_analysis.py`. You
   get running precision and recall, and you'll see if the estimate has already
   stabilised.
3. **If time runs short**, 120 labelled rows still gives a usable estimate with
   wider intervals. Report the n honestly. Zero rows gives nothing.

## What the numbers become

- **Precision** = share of retained rows judged `Y`. Expect roughly 70-80%
  based on cluster inspection.
- **Recall proxy** = share of excluded near-misses judged `Y`. Should be low;
  if it's high, the rule is too strict and that's worth reporting too.
- **Mention rate** = share of `is_about_class` = N. Feeds the caveat on the
  architecture share figures.

Report all three, plus the BORDERLINE count. A measured 74% with a stated
method beats an unmeasured claim of 90%.
