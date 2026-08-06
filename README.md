# witprobe-attention-memory

Artifacts, guards, and machine-checked proofs for the paper
**"Runtime Observability for Heterogeneous Attention Memory"** (Metask Lab).

This repository is built so that a reviewer can verify the paper at the
lowest possible cost: **every number, figure, and table in the paper
regenerates from the frozen run artifacts shipped here, and a claim guard
fails the build on any mismatch between manuscript and artifacts.**

## One-command reproduction

```bash
pip install -r requirements.txt   # matplotlib; L0 needs stdlib only
bash reproduce.sh                 # L0 numbers -> L1 figures -> L2 gates, single verdict at the end
```

Tested environment: Python 3.9.6 / matplotlib 3.9.4 on macOS and Linux;
no GPU, no network access, no model weights required. The Lean layer is
pinned by `formal/lean-toolchain` and run separately (see below).

## 30-second verification (no GPU, Python 3.9+ only)

```bash
python3 tools/make_canon.py          # regenerate the frozen-number canon from run artifacts
python3 tests/test_paper_claims.py   # every paper number must appear and match — or FAIL
```

If the second command prints `ALL PAPER CLAIM CHECKS PASSED`, every
quantitative claim in the paper traces to a JSON artifact in
`experiments/out/`.

## Layered reproduction

| Layer | What it verifies | Command | Needs | Time |
|---|---|---|---|---|
| L0 | every paper number ↔ artifact | `python3 tools/make_canon.py && python3 tests/test_paper_claims.py` | Python 3.9 | < 1 min |
| L1 | every figure & table regenerates | `python3 tools/p2_figs.py` | + matplotlib | < 1 min |
| L2 | the machine-decidable serving gates | `python3 experiments/flagship_gate.py` and `WITCERT_P98_PREFIX=p99 python3 experiments/p98_concurrent_identity.py` | Python 3.9 | < 1 min |
| L3 | all 67 Lean theorems, zero `sorry`, standard axioms only | `cd formal && bash check_all.sh` | elan + Mathlib cache | ~30 min first run |
| L4 | serving-level reruns | — | private probe platform + 8×HGX GPUs | see boundary below |

The exit code of each gate is the verdict; nothing is adjudicated by prose.

## What is here

- `experiments/out/` — the frozen artifacts behind every cited number
  (mechanically derived citation list; exploratory runs not cited by the
  paper are not included).
- `tools/` — the canon/figure/claim generators the paper is built from,
  plus the adjudication-data exporter.
- `tests/` — the claim guard and the flagship-gate regression tests.
- `formal/` — the complete Lean development: the certificate calculus,
  the observation-adequacy and key-identity rule families, and the
  experiment-adjudication calculus described in the paper.
- `witprobe/` — the measurement mathematics (`meters.py`) and the
  acceptance gate (`verify.py`): the parts of the platform that are
  themselves contributions of the paper.
- `papers/p2-attention-memory/` — the LaTeX sources; figures rebuild from
  `figdata.json`, whose numbers the claim guard checks against the prose.

## What is not here (release boundary)

The architecture-specific probe-injection implementation (hook anchors,
adapters, kernels, serving-stack patches) is not part of this release.
Serving-level reruns therefore require the private platform; the paper's
verifiable surface is the artifact-and-guard chain above, and the paper
states this boundary explicitly in its Limitations section.

## Requirements

- Python ≥ 3.9 (L1 additionally needs `matplotlib`)
- Lean 4 via `elan` for L3 (`formal/lean-toolchain` pins the version;
  first Mathlib build downloads the cache)

## License

Apache-2.0 (see `LICENSE`). If you use this artifact chain, please cite
the paper (arXiv id to appear).
