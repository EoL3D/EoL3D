# Question-answering tools

Scripts that check, convert, run and score the question-answer pack of the dataset
(`vlm/vlmpack_2026-11-05/` on Hugging Face). The pack ships the same tools under its own `tools/` folder.
Every command below runs from the root of this repository; `$DATA` is the folder the dataset was downloaded into
(see the top-level README) and `$OUT` is any folder for the outputs.

```bash
pip install -r vqa/requirements.txt          # PyYAML and Pillow, nothing else
PACK="$DATA/vlm/vlmpack_2026-11-05"
OUT=runs
```

The released evaluation subset is `$PACK/eval_v3/`: one file per unit (`scan`, `oak1`, `oak2`, `multi`), each with
its sidecars (`.images_manifest.jsonl`, `.oracle.jsonl`, `.summary.json`, `.masks.jsonl`). `$PACK/data/vlm_<unit>.jsonl`
are the full exports it was drawn from.

## Scripts

| script | for |
|---|---|
| `vlm_selfcheck.py` | writes the reply a perfect model would give to every record and scores it; every task must score 1.0 |
| `vlm_to_chat.py` | records -> chat requests in the OpenAI, Qwen-VL or InternVL format (marks drawn on copies of the pictures) |
| `vlm_run/vlm_run.py` | runs a model over a chat file: `vllm` or `transformers` for open models, `openai` for any OpenAI-compatible API, `gold` for the labels' own reply |
| `vlm_score.py` | grades the replies (strict and lenient parsing), per task, layer and template, with 95 % intervals from a bootstrap over desktops |
| `vlm_references.py` | the reference lines: the records without pictures (`blind`), with state tables instead of pictures (`state_oracle`), and a prior that answers from the previous action alone (`protocol_prior`) |
| `vlm_overlay.py` | the variant whose pictures carry the confirmed masks painted in class colours |
| `vlm_eval_set.py` | draws the evaluation subset from the full exports |
| `vlm_sample.py` | draws a stratified sample of one file: the human-audit sample or the repeat subset |
| `vlm_glossary.py`, `vlm_io.py` | the glossary a prompt ends with; reading the files and their sidecars |
| `vlm_run/run_eval.sh`, `vlm_run/collect_scores.sh`, `vlm_run/slurm/` | one evaluation job end to end, a table over all scores, Slurm templates |

`vqa/tda/` is the part of the annotation tool's library that the self-check and the scorer need (the taxonomy,
the state machine, the constraint graph and the grader `tda/core/export/vlm_check.py`);
`vqa/configs/taxonomy.yaml` is its vocabulary.

## 1. Check the pack

```bash
python vqa/vlm_selfcheck.py --truth $PACK/eval_v3/scan.jsonl $PACK/eval_v3/oak1.jsonl $PACK/eval_v3/oak2.jsonl \
    $PACK/eval_v3/multi.jsonl --out $OUT/selfcheck
```

It must end with `[vlm-selfcheck] OK`; it prints one line per file (`scan.jsonl: 1097 records, 5 task rows, all 1.0,
sidecars complete`, then 137 records for `oak1`, 66 for `oak2`, 4047 for `multi`).

## 2. Chat requests

```bash
python vqa/vlm_to_chat.py --export $PACK/eval_v3/oak2.jsonl --format openai --image-root $PACK/images \
    --out $OUT/oak2.openai.jsonl
```

`--format qwen` and `--format internvl` write the other two formats. A record whose parts are outlined in red gets a
marked copy of its picture, written beside the output (`$OUT/oak2.openai_marked/`) or under `--assets`.

## 3. Run a model

```bash
# the labels' own reply: a check of the chain, no model
python vqa/vlm_run/vlm_run.py --backend gold --chat $OUT/oak2.openai.jsonl --truth $PACK/eval_v3/oak2.jsonl \
    --out $OUT/gold.oak2.jsonl
# an open model with vLLM (pip install -r vqa/vlm_run/requirements.txt -r vqa/vlm_run/requirements-vllm.txt);
# --dry-run checks the inputs and prints the first prompt without loading anything
python vqa/vlm_run/vlm_run.py --backend vllm --model Qwen/Qwen3-VL-8B-Instruct --chat $OUT/oak2.openai.jsonl \
    --out $OUT/qwen.oak2.jsonl --dry-run
# a closed model through an OpenAI-compatible API: the key is read from the variable --api-key-env names
python vqa/vlm_run/vlm_run.py --backend openai --model "$API_MODEL" --api-key-env OPENAI_API_KEY \
    --chat $OUT/oak2.openai.jsonl --out $OUT/api.oak2.jsonl --dry-run
```

Drop `--dry-run` to run. The output has one line per request, `{"id", "raw_text", "meta"}`; a re-run skips the ids
already answered. Any other model works the same way as long as its replies end up in that format.

## 4. Score

```bash
python vqa/vlm_score.py --truth $PACK/eval_v3/oak2.jsonl --pred $OUT/gold.oak2.jsonl --chat $OUT/oak2.openai.jsonl \
    --out $OUT/scores --name gold-oak2
```

writes `$OUT/scores/gold-oak2.score.json` and `.score.md`. `--chat` maps boxes answered on a cropped or scaled picture
back to the frame.

## 5. Reference lines

```bash
python vqa/vlm_references.py --export $PACK/eval_v3/multi.jsonl --out $OUT/references
python vqa/vlm_score.py --truth $PACK/eval_v3/multi.jsonl --pred $OUT/references/protocol_prior.predictions.jsonl \
    --out $OUT/scores --name protocol_prior-multi
python vqa/vlm_to_chat.py --export $PACK/eval_v3/multi.jsonl --records $OUT/references/blind.jsonl --format openai \
    --out $OUT/multi.blind.openai.jsonl
```

The protocol prior comes with its own predictions; `blind.jsonl` and `state_oracle.jsonl` are records to send to a
model like any other (`--records`), scored against the plain truth file.

## 6. Draw a subset

```bash
python vqa/vlm_eval_set.py --export $PACK/data/vlm_scan.jsonl $PACK/data/vlm_oak1.jsonl $PACK/data/vlm_oak2.jsonl \
    $PACK/data/vlm_multi.jsonl --out $OUT/eval_v3 --audit 300
```

With the ids of the earlier subset in `configs/vlm_keep/` (at the top of this repository, read by default), this draw
reproduces the released `eval_v3/` byte for byte: `ids.txt` (5,347 ids), the four unit files and their manifest,
oracle and summary sidecars, `table.md`, `table.json`, and the 300-record audit sample in `audit/`.

The repeat subset (a model asked every question three times) is drawn per unit and kept in one file:

```bash
for unit in scan oak1 oak2 multi; do
  python vqa/vlm_sample.py --export $PACK/eval_v3/$unit.jsonl --out $OUT/repeat/$unit --purpose repeat
done
cat $OUT/repeat/*/repeat.ids.txt > $OUT/repeat.ids.txt
```

## 7. Evaluation jobs

`vlm_run/run_eval.sh` runs one unit end to end (subset, optional blind or overlay variant, chat requests, model,
scores); every setting is an environment variable, listed at the top of the script. With `TOOLS` it uses this
folder instead of the pack's `tools/`:

```bash
PACK=$PACK VIEW=oak2 BACKEND=gold TOOLS=$PWD/vqa RUNS=$OUT/runs bash vqa/vlm_run/run_eval.sh
PACK=$PACK VIEW=oak2 BACKEND=gold MODEL_TAG=gold_repeat SUBSET=$OUT/repeat.ids.txt REPEAT=3 TOOLS=$PWD/vqa RUNS=$OUT/runs \
    bash vqa/vlm_run/run_eval.sh
PACK=$PACK TOOLS=$PWD/vqa RUNS=$OUT/runs bash vqa/vlm_run/collect_scores.sh
```

The results go to `$OUT/runs/<model>/<variant>/<unit>/`, and `collect_scores.sh` writes `$OUT/runs/score.md`. On a
Slurm cluster, `vlm_run/slurm/open_model.sbatch` (vLLM on one GPU), `closed_model.sbatch` (an API, CPU only) and
`gold_check.sbatch` run `run_eval.sh` as an array over the units of the pack, and `submit_matrix.sh` submits every
model and variant; their headers give the `sbatch` lines, and the lines marked `EDIT:` are the ones to adapt to your
cluster.
