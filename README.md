# aGPT LongRoPE + ProLong-512K: original 2B and 20B to 128k

## Approved scope and current status

- Start independently from the original **2B** and **20B** weights, not YaRN-trained weights.
- Target **131072 tokens only**. No 16k/32k/64k training stages or cumulative chain.
- Use **princeton-nlp/prolong-data-512K** for continued pretraining.
- Jointly search per-frequency LongRoPE factors and the number of initial tokens
  retaining original RoPE. This is LongRoPE, **not LongRoPE2**.
- Full ProLong download/conversion is now approved. **No LongRoPE GPU jobs submitted.**

This folder was relocated to `../LongRoPE` from `../xpu_launcher/longRoPE`.
Use `.venv-data/bin/python` directly; activation scripts in the moved virtual
environment still contain the original path.
CPU-only preparation is submitted using `qsub job_prepare_data.sh` (capacity,
one node, up to 72 hours); it never submits training and writes
`experiments/prolong128k/data/PREPARED.json` only after complete validation.
Its log reports download, conversion and verification phases. Do not submit a
second copy or overwrite a failed conversion's `.pending` directory.

The requested training configuration is 32 nodes, TP4/CP4/FSDP24, GBS24,
400 steps per model (1,258,291,200 token exposures each). It is **not active**:
on 2026-11-06, PBS reported `capacity.resources_max.nodect = 16`, so capacity
cannot accept this 32-node request. The existing four-node execution defaults
below remain unchanged; do not start training until queue/node choices are
resolved and the full data is verified.

All new code, docs, dependencies, datasets, caches, profiles and results remain
under this directory. Original weights, the existing training environment and
evaluation harness are read-only inputs. Live YaRN training and evaluation
are not changed. The previous unsubmitted `experiments/post50/` is preserved,
but is incompatible with the new controller; do not start it.
[LEGACY_POST50.md](LEGACY_POST50.md) preserves its historical description.

The new root is [experiments/prolong128k](experiments/prolong128k).
Its `plan.json` describes two serial stages, 400 steps each, recovery DCP
every 10 steps, permanent DCP and verified HF exports at steps 50, 100, ...,
400: **16 exports**. These are execution defaults, not completed training.
There are no accepted searched profiles, trained LongRoPE checkpoints or
measured LongRoPE PPL/accuracy results yet.

## Retained-prefix optimization

For dimension pair `i`, the non-retained positions use
`inverse_frequency[i] = 1 / (factor[i] * theta ** (2*i/head_dim))`.
Original theta is preserved: 50000 for 2B, 500000 for 20B; both have
head dimension 128 and 64 factors. The extension ratio is 131072/8192 = 16.
The non-retained amplitude is `sqrt(1 + log(16)/log(8192))`.

For absolute document positions `p < retained_tokens`, **both frequency and
amplitude revert to original RoPE**: no interpolation and unit amplitude.
Positions are not shifted, and the rule is independent of CP partition and
prefill chunk boundaries. `0, 1, 4, 16, 64, 256` are searched jointly with
the frequency factors; zero is a valid control and may win. No nonzero
prefix length is falsely reported as optimized before search completes.
Default population 8 includes all six prefix choices; default generations 8,
seed 1234, two elites, monotone factors bounded between 1 and 16.
This is a bounded engineering search, not a claim to reproduce the paper's
full search budget, short-factor readjustment or 2M-token progressive recipe.
Short factors remain all ones.

Search calibration remains **PG19 validation**, four sufficiently long books,
separate from ProLong training and fixed ProLong held-out evaluation. Original
validation text is re-tokenized using each source model's tokenizer.
Each sample scores `target_length-2` tokens with one BOS/EOS likelihood-prefix
token and one reserved generation slot. Candidate document NLL is resumable;
calibration hashes, source fingerprint and search settings bind every profile.
Source fingerprints use weight file metadata, not full tensor-byte hashes.

### Runtime compatibility is mandatory

Stock HF/vLLM LongRoPE **does not implement retained tokens**. Merely adding
`retained_tokens` to standard `rope_scaling` would silently give wrong results.
This implementation uses:

- `runtime/.../models/common/longrope.py`: training caches, including the prefix.
- `runtime/modeling_longrope.py`: exported custom HF model, loaded with
  `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`.
- `runtime/longrope_vllm.py`: explicit
  `worker_cls=longrope_vllm.RetainedXPUWorker`, including spawned TP workers.
  The worker installs the retained-prefix cache implementation before model load.
- `runtime/longrope_profile.py`: writes retained-token metadata, custom HF code
  and `auto_map` into each isolated candidate view/export.
- `verify.py`: checks actual training caches against the custom HF implementation.

**Do not load these models with plain `LlamaForCausalLM` or a default vLLM
worker.** A stock loader can ignore the custom prefix metadata. Always use
the supplied runtime. No global Transformers/vLLM package is modified.
The custom HF model fixes the factor branch to the exported target cap to
avoid changing cached-key basis during generation. vLLM chooses short factors
for serving caps <=8192 and long factors above that threshold; record this
distinction in short-context comparisons.

## ProLong data and tokenizer safety

The upstream corpus has approximately 31B **Llama-3** tokens, not aGPT IDs.
It is MDS, not an HF `load_dataset` text repository. Even if source IDs fit
inside aGPT's vocabulary, passing them directly would train on wrong tokens.

`download_data.py` pins dataset revision
`0bb2becdbb958cf29804da01ae615eb838170aa4`. `--metadata-only` fetches README
and 12 indices only; without that flag it downloads the full, large corpus.
The metadata-only snapshot is explicitly rejected by conversion.

`prepare_data.py` uses `streaming.LocalDataset` and every `indices` pair
`[start, exclusive_end)` to decode separate documents with an explicitly
supplied local **Llama-3 tokenizer** (128256 vocabulary, BOS 128000).
Source special tokens are stripped, empty documents and invalid/gapped/
overlapping boundaries fail explicitly. Text is stored as checksummed parquet
and token counts are measured with each aGPT tokenizer. Training then encodes
that text with the normal aGPT tokenizer, adding its own BOS/EOS.
The recorded corpus counts exclude those added boundary tokens.

Before writing training parquet, conversion reserves a fixed approximately
1% of documents: SHA256 of the decoded UTF-8 text, interpreted as an integer,
modulo 100 equals zero. All copies of identical decoded text receive the same
assignment across all 12 subsets. Held-out copies are deduplicated, written
under `heldout/`, and **never** included in training streams or training token
counts. This is not near-duplicate detection or an official ProLong test split.
`manifest.json` records the policy, train/held-out counts and file checksums.
Old conversions lacking this policy must be regenerated, not relabeled.

All 12 subsets are required. The loader concatenates their training parquet
and applies a **global document permutation without replacement**. Each stage
derives its initial 32-bit seed from the SHA256 of
`stage_id|UTC first-training-start timestamp` (first eight hex digits).
The controller records the timestamp, stage and seed in the stage run's
`SAMPLING.json` before training, and forwards the same `AGPT_DATA_SEED` to
all ranks. Hourly continuation and failure retries reuse this record rather
than reading a new clock value. Existing checkpoints without their seed record
are rejected instead of silently changing the sampling order.
Arrow data and permutation indices are disk-backed, not a full text corpus
loaded into RAM. Every converted training row appears once per corpus epoch;
there is no weighted interleaver that re-loops smaller sources early.
Domain proportions follow the document counts, not equal domain weights.
After data-parallel partitioning, each rank reshuffles its partition at epoch
boundaries using seed `stage_initial_seed + epoch`. Checkpoint resume restores
the epoch, read position and token buffers, including the same shuffle, and
verifies that the saved seed matches the stage's persisted seed.
Duplicate training rows already present upstream remain distinct rows; the
no-replacement guarantee is per row, not near-duplicate content.
Each new context stage derives a time-dependent seed, so its initial ordering
is different in general but **not guaranteed disjoint** from earlier stages.
It retains the existing training packer/causal attention:
positions reset at document boundaries, but this is **not a reproduction of
ProLong's block-diagonal cross-document attention recipe**.
No PG19 fallback exists. `--max-records N` is smoke-only and its output is
rejected for training. Incomplete `.pending` conversions are not accepted.
Pre-submit validation checks tokenizer identity and every converted file hash.

The full source corpus and converted training corpus have **not** been
downloaded/prepared automatically. Large data preparation is an explicit step.
The MDS utility dependency is isolated in `.venv-data`; it is not installed
into the active training/evaluation environments.
MosaicML requires Transformers <5, so this data-only environment must **not**
be used for vLLM or training; those continue to use their existing environments.

## Commands: preparation, then review before GPU submission

Run from this directory after `module load frameworks/2026.1.0`.
The new experiment has already been prepared with:

```bash
/lus/flare/projects/datascience/seonghapark/venv/bin/python prepare.py --prepare-128k
```

Do not rerun it over the existing root. It never resets an experiment.

Data steps (CPU/network/disk only, potentially large):

```bash
# Inspect metadata first; remove --metadata-only only when ready for the full download.
.venv-data/bin/python download_data.py --metadata-only
.venv-data/bin/python download_data.py

# The real-data smoke check prepares this verified public Llama-3 tokenizer.
.venv-data/bin/python prepare_data.py \
  --source experiments/prolong128k/data/source \
  --source-tokenizer validation/mds_sample/tokenizer \
  --output experiments/prolong128k/data/prolong
```

`download_data.py` stores a revision receipt. A local source without that
receipt is not accepted. The upstream ProLong model's public tokenizer can be
used if verified as the same Llama-3 tokenizer; do not download model weights
just to obtain a tokenizer. Do not silently substitute another tokenizer if
access to the source tokenizer is unavailable.

Search is `debug`, one node/one hour, TP=1 for 2B or TP=4 for 20B.
Training/export is `debug-scaling`, four nodes/one hour, TP=4, CP=4,
FSDP=3 (48 ranks). Local batch and accumulation follow the existing launcher.
20B compile is disabled. The existing NaN watchdog, quarantine and HALT
policy remain; no unlimited retry is enabled. XPU memory/convergence
are not established by CPU tests.

**STOP for user approval before executing the following GPU-submit command:**

```bash
export LONGROPE_ROOT="$PWD/experiments/prolong128k"
export PYTHONPATH="$PWD/runtime"
/lus/flare/projects/datascience/seonghapark/venv/bin/python runtime/post50.py start
```

It validates full data and then submits a serial search -> train -> milestone
export chain, 2B followed by 20B. It refuses incompatible old plans and
unsearched profiles. Search profiles and training resume data must not be
overwritten; inspect `state.json`, `submissions.jsonl`, `HALT` and search logs.

## Subsequent YaRN ProLong sweep (not submitted)

`yarn_post50.py prepare` prepares the isolated
`experiments/yarn_prolong_post50` root. It reuses the existing YaRN controller,
launcher, export verification and NaN safeguards, with an explicit ProLong
dataset override; default/live PG19 runs are unchanged.
Both original models receive the full shared-16k, independent-32/64/128k,
cumulative-32/64/128k sweep: **14 stages, 400 steps/stage, 112 HF exports**.
Cumulative stages inherit this sweep's preceding YaRN weights, not LongRoPE
weights. Training uses the same fixed held-out exclusion and non-replacement
shuffle and time-dependent persisted stage seeds as LongRoPE.
Existing batch/TP/CP settings are preserved (including
the 20B 128k token-budget difference documented for the PG19 experiment).

No automatic connection submits this sweep after LongRoPE. A human must
approve GPU submission after LongRoPE's root `DONE` and all 16 verified
exports exist, and after the full split corpus passes checksums/tokenizer
validation:

```bash
# Preparation only; already performed, do not overwrite the prepared root.
/lus/flare/projects/datascience/seonghapark/venv/bin/python yarn_post50.py prepare

# Only after approval; this command SUBMITS a GPU job.
/lus/flare/projects/datascience/seonghapark/venv/bin/python \
  yarn_post50.py start --approve-gpu
```

Jobs use debug-scaling, 4 nodes, 1 hour, with the original serial
train/export continuation. Current PG19 models/results and LongRoPE profiles
are not overwritten. Source data is shared read-only via a symlink.
The YaRN controller copies the training shell into per-job logs before
execution, so edits to the shared launcher cannot change a running shell's
later commands.

## Evaluation and validation

After all 16 exports are verified, `evaluation/prepare.py EXPERIMENT_ROOT`
prepares 54 plans / 1746 shards, **without submission**:
original baseline + eight milestones per size, standard 8 tasks,
LongBench 4, RULER 13, MMLU 57 subjects/5-shot, fixed ProLong held-out
token PPL at 4k/8k/16k/32k/64k/128k. Each baseline is unchanged.
Token PPL is `exp(total_NLL / total_target_tokens)`, never the average
of document perplexities. The task is `prolong_ppl` (108 PPL shards).
YaRN uses the same held-out partition through the shared corpus preparer;
each tokenizer's counts and the manifest checksum are recorded. Short
documents do not exercise a full 128k context; corpus statistics include
the number of documents at least 128k tokens long. Evaluation also requires approval.

Validation commands:

```bash
export PYTHONPATH="$PWD/runtime:$PWD/runtime/xpu_torchtitan/torchtitan_repo"
/lus/flare/projects/datascience/seonghapark/venv/bin/python \
  -m unittest test_longrope test_retained_prolong
/lus/flare/projects/datascience/seonghapark/llm_evaluation/.venv-post2b/bin/python \
  -m unittest test_vllm_longrope
```

Reports are under `validation/`. Synthetic profiles, tiny random HF weights
and MDS fixtures in CPU tests are **not measured aGPT results**.
Real GPU search, distributed training and inference remain unexecuted.
`validation/STATUS.json` records 18 training/export/data tests, two actual-vLLM
CPU tests, both launcher dry runs and the successful real-record check.
`validate_data_sample.py` is an additional CPU-only real-data smoke check:
it reads one record from the first upstream book MDS shard using a public
ProLong/Llama-3 tokenizer and records counts, not raw book text, under
`validation/mds_sample/RESULT.json`. This is not a full training conversion.
CPU-only launcher dry runs use `NPROC_PER_NODE=12`, 48 simulated ranks and
a stand-in corpus token count. They exercise argument forwarding, not XPU
memory or distributed execution.

## Sources

- [LongRoPE paper (ICML 2024)](https://arxiv.org/abs/2402.13753)
- [Official retained-token implementation, pinned](https://github.com/microsoft/LongRoPE/blob/3bf9249acd651fab055903e43aa5bb4f6514c4c7/rope/longrope.py)
- [ProLong-512K dataset](https://huggingface.co/datasets/princeton-nlp/prolong-data-512K)
- [ProLong document indices / training loader, pinned](https://github.com/princeton-nlp/ProLong/blob/499fa2994fa727b4c8b53b706ccd5c23f77d7f89/training/dataset.py)
- [ProLong paper](https://arxiv.org/abs/2410.02660)

Algorithms were implemented locally; Microsoft code was inspected, not copied
into this workflow. The existing TorchTitan snapshot retains its license.
