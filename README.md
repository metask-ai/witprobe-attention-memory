# witprobe-attention-memory

Artifacts, guards, and machine-checked proofs for the paper
**"Runtime Observability for Heterogeneous Attention Memory"** — Fanzhe Wei, Li Liu, Ziyang Wang, Chenyu Wang,
[arXiv:2608.05863](https://arxiv.org/abs/2608.05863).

## The four-paper series

| | Paper | Paper link | Artifact |
|---|---|---|---|
| **P1** | WitCert: Sound Runtime Risk Observability and Gating for KV-Cache Quantization | [arXiv:2607.28699](https://arxiv.org/abs/2607.28699) | [witcert-kv-certificates](https://github.com/metask-ai/witcert-kv-certificates) |
| **P2** | Runtime Observability for Heterogeneous Attention Memory | [arXiv:2608.05863](https://arxiv.org/abs/2608.05863) | [witprobe-attention-memory](https://github.com/metask-ai/witprobe-attention-memory) ← **this repository** |
| **P3** | Pricing the Risk of Runtime Compression: Anytime-Valid Admission and a Served-Output Law for Compressed Serving State | [arXiv:2608.15810](https://arxiv.org/abs/2608.15810) | [witcert-w-certified-precision](https://github.com/metask-ai/witcert-w-certified-precision) |
| **P4** | What to Protect When You Quantize a Mixture of Experts | not yet public | not yet public |

These four papers are one line of work, not four topics. **P1** asks whether
compression is damaging *the request being served right now*, and answers it
for the KV cache with a provably sound runtime meter and meter-driven gating.
**P2** carries the same question to the fact that a modern model's memory is no
longer a plain KV cache — latent caches, learned sparse selectors and recurrent
states each fail differently — and gives one observability contract for all
four classes. Together they answer *whether it can be measured*. **P3** asks
what the measured risk is worth and how to spend it: the union budget those
systems rely on exhausts on every long production request, and what replaces it
is an anytime-valid ledger, a law carrying the certified witness to the served
output, and the quantifier that makes the bound hold on a request never seen.
**P4** asks the converse — *what that machinery should be pointed at* — and
prices the field's shared instinct that MoE routing invariance must be
protected, finding it wrong in three independent ways.

Each paper stands alone: P3 and P4 inherit the typed-contract vocabulary of P1
and P2 but restate no result of theirs, and neither claims the other's.

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
the paper:
[arXiv:2608.05863](https://arxiv.org/abs/2608.05863);
machine-readable metadata is in `CITATION.cff`.
