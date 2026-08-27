# Unattended wake word training

A start-to-finish pipeline that turns a phrase into a `.tflite` micro-wake-word
model, designed around one constraint: **you should be able to start it, close
your laptop, and come back to a finished model.**

It is the same underlying training recipe as the notebook in the repo root,
restructured so that nothing depends on a browser tab staying open.

---

## Why not just use the notebook?

The notebook works, but it is a notebook. Every heavy step lives in a cell, and
a cell only runs while something is holding the session open. On free Colab that
means a disconnect after ~90 minutes of inactivity, a hard cap around 12 hours,
and no way to resume — a session that dies during hour four of training leaves
you with nothing to restart from.

This pipeline fixes that structurally rather than by babysitting:

| Problem | How this handles it |
|---|---|
| Session dies mid-run | Every step writes a completion marker. Re-running skips finished work. |
| Training dies at 30k of 45k steps | Training resumes from its last checkpoint automatically. |
| Half-written feature files look complete | A partial mmap is detected and deleted rather than silently trained on. |
| Sample generation dies at 35k of 50k clips | Existing clips are counted; only the shortfall is generated. |
| You can't tell what it's doing | `./status.sh` prints a short summary you can read or paste anywhere. |
| Machine reboots or gets preempted | `--service` installs a systemd unit that restarts on boot. |

The practical consequence: the worst a crash can cost you is the step that was
running, not the whole run.

---

## Where to run it

You need a Linux machine with an NVIDIA GPU for a few hours. Ranked for your
situation:

### 1. A GCP VM using your $300 trial credits — recommended

This is what the credits are good for, and it is the only option on this list
that genuinely runs unattended. You start the VM, start the pipeline, close
everything, and check back later. Expect **roughly $6–10 of the $300** for a
complete run.

Two things to know before you start:

- **GPUs are blocked on a Free Trial billing account.** You must upgrade to a
  paid account first. Upgrading does *not* consume or forfeit your credits —
  they carry over and are still spent before you are charged anything. ([Google's
  docs][gcp-free]) You then request a GPU quota increase under
  *IAM & Admin → Quotas* (ask for 1 of `GPUS_ALL_REGIONS`). Approval is usually
  quick but is not always instant, so do this first.
- **Credits expire 90 days after you signed up**, not 90 days from first use. If
  you signed up a while ago, check what's actually left before planning around
  it.

### 2. RunPod or Vast.ai — the fast path if GCP quota is a hassle

Rent a GPU by the hour with no quota request and no account upgrade. A run costs
about **$3–6** total. You get a plain Ubuntu box with drivers already installed,
so `bootstrap.sh` works unchanged. This is the lowest-friction option and I'd
take it over fighting GCP quota — the only reason it isn't first is that you
already have the credits.

### 3. Colab, driven by the Colab CLI — free, and better than it sounds

Colab's CLI provisions a runtime from your terminal and runs a **keep-alive
daemon** that holds it with no browser tab open — the same mechanism Google's
own VS Code extension uses, so this is a supported path rather than a trick.
Combined with `--archive-dir` (below), a reclaimed runtime costs you one step
rather than the run.

A free-tier account does get a T4 this way. What you don't get is an unlimited
session: the runtime is still reclaimed on Colab's schedule, so a full run
usually spans two sessions. `colab/colab_run.sh` drives the whole thing —
see **Running it on Colab** below.

**What I'd do:** start on Colab, since it costs nothing and needs no approval.
Request GCP quota in parallel; if it lands, move over for a single
uninterrupted run.

[gcp-free]: https://docs.cloud.google.com/free/docs/free-cloud-features

---

## GCP walkthrough

After upgrading to a paid account and getting GPU quota:

```bash
gcloud compute instances create wakeword-trainer \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu121-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True"
```

The Deep Learning VM image ships with CUDA and the NVIDIA driver, which saves a
fussy install. If that image family name has moved, list what's current with
`gcloud compute images list --project deeplearning-platform-release | grep cu12`.

200 GB is deliberate. The generated clips, augmentation audio and spectrogram
features together run roughly 30-50 GB; the rest is headroom, and disk is the
cheapest part of this whole exercise.

Then:

```bash
gcloud compute ssh wakeword-trainer --zone=us-central1-a

git clone https://github.com/trout1758-cpu/custom-micro-wake-word-model
cd custom-micro-wake-word-model/pipeline
./bootstrap.sh
```

> **Stop the VM when the run finishes.** A stopped VM costs only its disk (a few
> cents a day); a running idle GPU VM burns roughly $20/day of your credits.
> `gcloud compute instances stop wakeword-trainer --zone=us-central1-a`

---

## Running it

### Step 1 — check the pronunciation first

```bash
./run.sh config/fbi_guy.yaml --preview
```

This writes a handful of clips to `~/wakeword-work/preview/`. Copy them back and
listen:

```bash
gcloud compute scp --recurse \
  wakeword-trainer:~/wakeword-work/preview ./preview --zone=us-central1-a
```

They should sound like a clear "eff-bee-eye guy". **Do not skip this.** Piper is
fed IPA phonemes rather than the text "FBI guy" precisely because plain text is
a coin flip on acronyms — some voices say "fibby", some spell it out, some drop
a letter. If the preview is wrong, fix `wake_word.phonemes` in the config and
re-preview. Every hour after this point is spent teaching the model whatever is
in those clips.

### Step 2 — start the real run

```bash
./run.sh config/fbi_guy.yaml --service
```

`--service` installs a systemd unit, so the run survives logging out, the VM
rebooting, and the process being killed. Now close the terminal.

(`--detach` is the lighter version — survives logout but not a reboot.
No flag at all runs it in the foreground, which is useful for debugging.)

### Step 3 — check on it whenever

```bash
./status.sh              # short summary
./status.sh --watch      # refresh every 30s
./status.sh --log        # follow the live log
```

`./status.sh` output is deliberately compact — it's what to paste if you want
help reading a failure.

### If something breaks

Run the same command again. Finished steps are skipped, training resumes from
its checkpoint. That is the entire recovery procedure.

To force one step to re-run:

```bash
./run.sh config/fbi_guy.yaml --redo 03_features
./run.sh config/fbi_guy.yaml --only 04_train     # just this one
./run.sh config/fbi_guy.yaml --from-step 04_train
```

---

## Running it on Colab

Colab needs no account upgrade, no quota request, and no money. The tradeoff is
that the runtime gets reclaimed on Colab's schedule, so the run normally spans
two sessions. The archive is what makes that cheap instead of painful.

### Once

```bash
uv tool install google-colab-cli     # or: pipx install google-colab-cli
```

If `colab` is then "not found", it is a PATH problem, not a failed install —
`uv` puts tools in `~/.local/bin`.

Authentication is a copy-paste flow by default: it prints a URL, you approve it
in a browser on any machine, and paste the code back. Nothing needs a display on
the machine running it, so a headless box is fine.

### Every session

```bash
./colab/colab_run.sh setup      # provision a T4, mount Drive, install everything
./colab/colab_run.sh preview    # generate clips and download them — listen first
./colab/colab_run.sh start      # launch detached, then close the terminal
```

Then check in whenever:

```bash
./colab/colab_run.sh status
./colab/colab_run.sh log
```

When it finishes:

```bash
./colab/colab_run.sh fetch      # downloads the .tflite and .json
./colab/colab_run.sh stop       # release the runtime
```

### When Colab takes the runtime away

```bash
./colab/colab_run.sh setup && ./colab/colab_run.sh start
```

That is the whole recovery. Setup restores the completed steps from the Drive
archive instead of regenerating them, and training resumes from its last
mirrored checkpoint.

### Watch the disk

`setup` prints free space under `/content` and warns if it looks tight. The run
needs roughly 30-50 GB. If your runtime has less, lower `positives.max_samples`
to `25000` and `datasets.audioset_clips` to `8000` in the config — the model
will be somewhat worse on false accepts, but it will finish.

---

## Surviving an ephemeral machine

`--archive-dir` mirrors each finished step to durable storage — a mounted Drive,
a network share, a second disk — so a machine that disappears does not take the
run with it.

```bash
./run.sh config/fbi_guy.yaml --detach --archive-dir /content/drive/MyDrive/wakeword-archive
```

or set `WAKEWORD_ARCHIVE_DIR`. `colab_run.sh` does this for you.

What it does:

- **After each expensive step**, tars that step's output into the archive.
- **On startup**, restores anything missing locally, then marks those steps done.
- **During training**, mirrors checkpoints every five minutes — training is one
  long step, so waiting until it finishes would mean losing hours to a runtime
  that vanishes at hour four.

Two deliberate choices worth knowing:

- **Step 2 is not archived.** Those datasets are plain downloads; fetching them
  from the original hosts again costs about what a Drive round-trip costs.
- **A step's `.done` marker is only restored if its data was.** Restoring a
  marker whose artifacts failed to come back would make the pipeline skip a step
  whose output does not exist, and the failure would surface somewhere far less
  obvious. Local data always wins over the archive; nothing is overwritten.

On a persistent machine you do not need any of this — the work directory is
already durable. Leave the flag off.

---

## What the steps do, and how long they take

Times are for a single NVIDIA T4.

| Step | What happens | Time |
|---|---|---|
| `01_samples` | Piper generates 50k clips of the wake word and 1k each of 23 confusable phrases | ~40 min |
| `02_datasets` | Downloads room impulse responses, AudioSet ambient audio, music, and four pre-computed negative feature sets | ~1 hr |
| `03_features` | Augments every clip and converts it to 40-band spectrograms | 1–2 hr |
| `04_train` | Trains 45k steps in two phases | 4–8 hr |
| `05_export` | Copies out the `.tflite` and writes the ESPHome manifest | seconds |

**Total: roughly 8–12 hours**, unattended.

### The three kinds of training data

Understanding this is most of what you need to tune the model:

- **Positives** — TTS clips of "FBI guy". Teach it what to fire on.
- **Confusable negatives** — clips of phrases that sound close but must *not*
  fire. These carry a high penalty weight and are the biggest single lever on
  false triggers.
- **Ambient negatives** — pre-computed features of conversation, TV-like audio
  and background noise, from the microWakeWord project. These are what keep the
  model quiet in a real room.

---

## Why these confusable phrases

The model does not learn "F, then B, then I". It learns a vowel contour:

```
   FBI guy   ->   /ɛ/   /iː/   /aɪ/   /ɡaɪ/
                  eff   bee    eye    guy
```

Anything with that contour is a false-trigger risk, and the consonants matter
much less than the vowels. So the confusable list is built systematically —
substituting each position with other letters that share its vowel:

- `/ɛ/` letters: **F S L M N X** (eff, ess, ell, em, en, ex)
- `/iː/` letters: **B D G P T V C E Z** (bee, dee, gee, pee, tee, vee…)
- `/aɪ/` letters: **I Y** (eye, why)

That produces `N B I guy`, `S T I guy`, `L G I guy`, `F D I guy`, `F B Y guy`
and so on — plus truncations (`F B I` with no "guy", `B I guy` with no F) and
the bare `/aɪ … aɪ ɡaɪ/` tail (`wifi guy`, `sci fi guy`).

Unrelated acronyms like "TV guy" or "CIA guy" are deliberately excluded. Their
contours are far enough away that training on them would spend capacity without
buying much. If the finished device ever trips on one, add it to the config and
retrain — it's a two-line edit.

---

## Tuning

Everything is in `config/fbi_guy.yaml`. Nothing requires editing code.

| If the model… | Change |
|---|---|
| Triggers on random speech or TV | Raise `training.negative_class_weight`, e.g. `[60, 75]` |
| Triggers on near-miss phrases | Add them to `confusables.phrases` and retrain |
| Doesn't fire when you actually speak | Lower `negative_class_weight`, or raise `positive_class_weight` |
| Fires too easily in the real room | Raise `manifest.probability_cutoff` — no retrain needed, just edit the `.json` |
| Runs out of GPU memory | Lower `training.batch_size` to 128 |

Watch `estimated false positives per hour` in the log as training progresses —
it should trend toward `target_false_accepts_per_hour` during phase one.

To train a second version without destroying the first, bump `wake_word.version`.
Artifacts are named `fbi_guy_v1`, `fbi_guy_v2`, and so on, so you can flash both
to the device and compare.

### Adding your own voice

Optional, but it measurably improves real-world reliability: record 100–200 clips
of yourself saying the wake word, drop the `.wav` files into
`~/wakeword-work/real_recordings/`, and re-run. The pipeline picks them up
automatically and mixes them into training alongside the synthetic clips. Cells
10–12 of the notebook in the repo root handle the recording if you want a
ready-made way to do it.

---

## Deploying

When the run finishes, `~/wakeword-work/output/` holds two files:

```
fbi_guy_v1.tflite     the model
fbi_guy_v1.json       the ESPHome manifest
```

Copy both back and into your ESPHome config directory:

```bash
gcloud compute scp --recurse \
  wakeword-trainer:~/wakeword-work/output ./output --zone=us-central1-a
```

Then reference it from your device YAML:

```yaml
micro_wake_word:
  models:
    - model: fbi_guy_v1.json
```

`probability_cutoff` in the JSON is a deployment-time threshold, not part of the
model — raise it if the device is too trigger-happy, lower it if it's deaf. You
can iterate on that value without retraining anything.

---

## Layout

```
pipeline/
  config/fbi_guy.yaml      everything configurable, in one file
  bootstrap.sh             one-time machine setup
  run.sh                   start the pipeline (--preview / --detach / --service)
  status.sh                progress summary
  colab/colab_run.sh       drive the whole thing on a Colab runtime
  wakeword/
    config.py              config parsing and the work-directory layout
    state.py               step markers and status.json — the resume machinery
    archive.py             mirroring to durable storage for ephemeral machines
    piper.py               resumable TTS generation
    steps/                 the five pipeline steps
  tests/                   tests for the parts that don't need a GPU
```

Run the tests with `python -m pytest tests/ -q` (needs only `pyyaml` and
`pytest`; they do not touch the GPU or the network), plus
`./tests/test_colab_driver.sh` for the Colab driver's remote-execution shim,
which runs against a stub CLI.

---

## Attribution

The training framework, model architecture and negative datasets are
[microWakeWord](https://github.com/kahrendt/microWakeWord) by
[@kahrendt](https://github.com/kahrendt). This directory is orchestration around
it. See the repo root README for the notebook this pipeline was derived from and
the patches applied to upstream.

Training data has mixed licenses — models produced here are for non-commercial
personal use.
