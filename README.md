# megaDNA-endolysins

A practical pipeline around **megaDNA** (a hierarchical Transformer genomic LM) for:

- **Pretraining** on hundreds of thousands of bacteriophage genomes *(training data not included)*  
- **Knockout tests** to probe whether the model assigns higher loss / lower (log)likelihood to genomes with essential-gene KOs  
- **Fine-tuning** on coding regions (endolysins)  
- **Protein generation** from DNA via beam search scored by a **codon-usage–based** heuristic  
- **Protein analysis** utilities (see `protein_analysis/`)

> ⚠️ **Tokenizer variants**  
> The code supports an “old” 8-token setup and a “new” 12-token setup (with protein delimiters).  
> **Keep the setting consistent** across pretraining → fine-tuning → generation.

---

## Table of Contents

- [Install](#install)
  - [System tools (for knockout & pangenome tests)](#system-tools-for-knockout--pangenome-tests)
  - [Python (via `requirements.txt`)](#python-via-requirementstxt)
- [Data & configs](#data--configs)
- [Usage](#usage)
  - [1) Pretraining — `train.py`](#1-pretraining--trainpy)
  - [2) Fine-tuning on coding regions — `finetuning.py`](#2-fine-tuning-on-coding-regions--finetuningpy)
  - [3) Protein generation — `generate_endolysins.py`](#3-protein-generation--generate_endolysinspy)
  - [4) Knockout tests](#4-knockout-tests)
    - [a) Essential-hit (λ-like)](#a-essential-hit-λ-like)
    - [b) Pangenome-frequency](#b-pangenome-frequency)
- [Protein analysis](#protein-analysis)
- [Tips & gotchas](#tips--gotchas)
- [License](#license)

---

## Install

### System tools (for knockout & pangenome tests)

- **Prokka** (genome annotation)  
- **NCBI BLAST+** (`makeblastdb`, `blastp`)  
- **Roary** (pangenome; may require `cd-hit`, `mafft`, `parallel`, etc.)

Example (Ubuntu-like):

```bash
sudo apt-get update
sudo apt-get install -y prokka ncbi-blast+ roary cd-hit mafft parallel
pip install -r requirements.txt
```

## Data & configs

Training data (phage genomes) are not included.

Config YAMLs live in config/
(default files: config.yaml for pretraining, configfinetuning.yaml for fine-tuning).


## Usage
### 1) Pretraining — train.py

Pretrains megaDNA with DDP on GPUs (or multi-process CPU).

Args (key):
```bash
-i/--input FASTA, -o/--output dir, -p/--platform {CUDA,CPU}, -x/--gpu_index "0,1,...",
-c/--config YAML (default config.yaml), -t/--tokenizer {old,new}.
```

Outputs (in -o):

checkpoints/checkpoint_{E}.pth (state dicts)

splits/{train,val,test}_dataset.pth

Losses.png, Accuracies.png, Losses_steps.png, learning_rate.png

*.txt logs for losses/accuracies/lr

### 2) Fine-tuning on coding regions — finetuning.py

Fine-tunes from a chosen pretraining checkpoint on endolysin coding DNA
(adds optional upstream context with the “new” tokenizer).

Args (key):
```bash
-i/--input FASTA (endolysin sequences with context),
-f/--foldertofinetune pretrain folder (contains checkpoints/),
-n/--checkpointnumber which pretrain checkpoint to load,
-o/--output out dir,
-p/--platform + -x/--gpu_index,
-c/--config YAML (default configfinetuning.yaml),
-t/--tokenizer {old,new},
-k/--nkeep nucleotides to keep upstream.
```

Outputs (in -o):

checkpoints_finetuning/checkpoint_{E}.pth

Updated plots/logs (loss/acc/lr), splits for fine-tuning

### 3) Protein generation — generate_endolysins.py

Generates coding DNA from a primer (context + first coding chunk) using beam search in nucleotide-triplet blocks, scoring candidates by codon-usage RMSD

Args (key):
```bash
-i/--input folder with checkpoints (from fine-tune),
-n/--checkpointnumber,
-o/--output file name for output FASTA (saved in a subfolder),
-p/--platform + -x/--gpu_index,
-t/--tokenizer {old,new},
--primer_fasta FASTA with nucleotide templates,
--primer_length coding primer length,
--context number of context nt to keep,
--tpc/--total_prepended_context total nt of context present in the FASTA,
--tot/--total_to_generate total tokens to generate,
--topk top-k filtering threshold, --temp temperature,
--bsize block size (tokens per step), --repfactor repetition penalty factor.
```

Example (CUDA 0):


### 4) Knockout tests
#### a) Essential-hit (λ-like) — test_phagelambda.py / test_knockout_essentials.py

Annotate with Prokka, BLASTp proteins against your essential-proteins DB, pick best hit by identity × coverage, perform KO and compare wild_type vs essential_KO vs hypothetical_KO by loss / (log)likelihood.

Prepare the BLAST DB once:
```bash
makeblastdb -in Data/essentialproteins/essentials_proteins.faa \
  -dbtype prot \
  -out Data/essentialproteins/essentials_portal_terminase_major_db
```


Run:
'''bash
python test_phagelambda.py
python test_knockout_essentials.py
'''

Outputs:

Per-sequence Prokka outputs

results.csv (loss, likelihood, loglikelihood per condition)

Histograms: loss_distribution.png, likelihood_distribution.png, loglikelihood_distribution.png

#### b) Pangenome-frequency — test_pangenome.py

Annotate many genomes with Prokka, run Roary to compute the pangenome, then for each genome KO the most-represented vs least-represented gene groups; compare wild_type vs most_rep_KO vs least_rep_KO by loss/likelihood.

Run:
'''bash
python test_pangenome.py
'''

Outputs:

Prokka annotations, Roary outputs (e.g., clustered_proteins, gene_presence_absence.csv)

results.csv, distribution plots

### Protein analysis

protein_analysis/ contains several scripts to assess generated proteins using external tools and heuristics.
They are modular and can be run independently; see the in-file docstrings or comments for per-tool usage. (Details omitted here for brevity.)

### Tips & gotchas

Keep tokenizer (old/new) consistent across pretrain → fine-tune → generation.

For GPU training, prefer even batch/accumulation values so accuracy sampling aligns with optimizer steps.

Prokka/BLAST/Roary are CPU-intensive — you may want to subsample test sets for quick iterations.

The generation script’s scoring is illustrative; bring your own codon usage stats and adjust weights if needed.
