# PromLoop
Integrating Pretrained Biological Representations for Identification and Generation of Industrial Microbial Promoters.

## 📁 Datasets

We have constructed high-quality and cross-species promoter datasets covering six representative industrial microorganisms. All promoter sequences are length-normalized to 81 bp (from –60 to +20 bp relative to the transcription start site), covering the core regulatory region. The dataset is strictly deduplicated (CD-HIT, 90% identity) and split into training / validation / test sets with a 7:1:2 ratio, maintaining an approximately 1:1 positive-negative balance.

👉 **All dataset files are available in the [`Datasets/`](./Datasets/) directory**, organized by species (e.g., `Escherichia coli/`, `Bacillus subtilis/`, etc.).

## 🧬 Model Usage

Our fine-tuned promoter identification models for six industrial microbial species are built upon the **LucaOne** biological foundation model. To perform inference with these models (i.e., to identify promoters in your own sequences), please refer to the official **LucaOneTasks** project, which provides a complete pipeline for loading fine-tuned LucaOne models and running predictions.

👉 **Inference Guide**: [https://github.com/LucaOne/LucaOneTasks](https://github.com/LucaOne/LucaOneTasks)

Follow their README to:
- Load the fine-tuned model from Hugging Face (our model IDs are listed below)
- Run promoter identification on your sequences
- Obtain output CSV files similar to the ones we provide in this repository (e.g., `E.coli_prediction_results.csv`)

### Model Zoo

All six fine-tuned models are available on Hugging Face Hub:

| Target Species | Hugging Face Model ID |
| :--- | :--- |
| *Escherichia coli* | [`lucaone-ecoli-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-ecoli-promoter-id) |
| *Bacillus subtilis* | [`lucaone-bsutilis-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-bsubtilis-promoter-id) |
| *Diphtheriae* | [`lucaone-diphtheria-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-diphtheria-promoter-id) |
| *Baumannii* | [`lucaone-baumanii-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-baumanii-promoter-id) |
| *Staphylococcus* | [`lucaone-staphylococcus-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-staphylococcus-promoter-id) |
| *Bradyrhizobium* | [`lucaone-bradyhizobium-promoter-id`](https://huggingface.co/huangzhengsheng/lucaone-bradyrhizobium-promoter-id) |


## 🧬 Promoter Generation with LucaVAE

Beyond promoter identification, we provide a **conditional variational autoencoder (LucaVAE)** for *de novo* generation of functional promoter sequences. LucaVAE leverages the same LucaOne embeddings as conditional priors, enabling function‑guided sequence synthesis.

👉 The generation code is available in [`LucaVAE.py`](./LucaVAE.py).

### How to Use LucaVAE

#### 1. Prepare Input Data

You need two things:
- A CSV file containing promoter sequences (e.g., `Datasets/Escherichia coli/Datasets.csv`) with a column named `seq` (or customize `SEQUENCE_COLUMN_NAME`).
- Pre‑computed LucaOne embeddings for each sequence (as `.pt` files).  
  *See the [LucaOneTasks](https://github.com/LucaOne/LucaOneTasks) repository for how to generate embeddings.*

#### 2. Configure and Run

Edit the configuration section in `LucaVAE.py`:

```python
CSV_FILE_PATH = "path/to/your/sequences.csv"
EMBEDDING_DIR = "path/to/your/embedding/files"
SEQUENCE_COLUMN_NAME = "seq"
MATRIX_FILE_PATTERN = "matrix_{}.pt"   # adjust to your file naming
CHECKPOINT_DIR = "./lucavae_checkpoints"
```

Then run the script:
```python
python LucaVAE.py
```

The script will:
- Train the VAE for up to 200 epochs (early stopping on validation loss).
- Save the best model checkpoint and final model weights.
- Generate new promoter sequences using the best model (default: 5 sequences per condition embedding, temperature = 1.2).
- Output a CSV file (generated_promoters_<N>.csv) with unique generated sequences.



## 📊 Experimental Results – Reproducibility Data

To ensure full transparency and reproducibility of all results reported in our paper, we provide the complete prediction outputs for **five different experimental configurations** on the test sets of all six microbial species.

Each CSV file contains:
- `seq_id`: sequence identifier
- `seq`: 81 bp promoter sequence
- `prob`: predicted probability (0–1) of being a functional promoter
- `label`: ground truth (1 = promoter, 0 = non‑promoter)

👉 All result files are organized in the [`Results/`](./Results/) folder.


These results correspond to the following experiments in our paper:

| Folder | Paper Reference | Description |
| :--- | :--- | :--- |
| `1_lucaone_lucabase_ft50` | Table 2, Table 3, Table 4 | Our proposed method (LucaOne + LucaBase, fine-tuned 50 epochs) |
| `2_dnabert2_lucabase_ft50` | Table 2 | Baseline: DNABert2 embedding + LucaBase |
| `3_onehot_lucabase_ft50` | Table 2 | Baseline: Onehot encoding + LucaBase |
| `4_lucaone_linear_ft50` | Table 3 | Comparison: LucaOne + linear classifier |
| `5_lucaone_lucabase_zero_shot` | Table 4 | Zero‑shot transfer (no fine‑tuning) |

You can directly load any CSV file to reproduce our evaluation metrics (Accuracy, Precision, Recall, F1, MCC, AUC) or to compare with your own models.


## 🔬 Evaluation Tools

To help you reproduce all quantitative analyses in our paper or apply the same evaluation pipeline to your own data, we provide three standalone Python scripts in the [`Evaluation/`](./Evaluation/) folder.

| Script | Purpose | Corresponding Paper Section |
| :--- | :--- | :--- |
| `Identification_metrics.py` | Compute Accuracy, Precision, Recall, F1, MCC, AUC from prediction CSV files | Section 4.2 (Tables 2–5) |
| `GC & Kmer.py` | Generate GC content boxplots and 3‑mer frequency bar plots for generated vs. real sequences | Section 4.2.6 (Figure 4 & Appendix A) |
| `motif_analysis.py` | Extract top enriched 6‑mers and count known core promoter motifs (TATA‑box, DPE, GC‑box) | Section 4.2.7 (Figure 5 & Table 7) |

### 1. Identification Metrics (`Identification_metrics.py`)

**Usage** – Modify the file paths at the bottom of the script:

```python
metrics, true_labels, pred_labels = load_and_calculate(
    '../Datasets/Bradyrhizobium/test.csv',          # ground truth file
    '../Results/1_lucaone_lucabase_ft50/Bradyrhizobium.csv',  # prediction file
    true_label_col='label',
    pred_label_col='label',
    pred_prob_col='prob'
)
print_metrics(metrics)
```

Run the script:
```python
python Evaluation/Identification_metrics.py
```

### 2. GC & 3‑mer Analysis (`GC & Kmer.py`)
**Usage** – Edit the configuration section at the top of the script:

```python
GENERATED_PREDICTIONS_FILE = "../Results/1_lucaone_lucabase_ft50/Bradyrhizobium.csv"
REAL_SAMPLES_FILE = "../Datasets/Bradyrhizobium/positive_samples.csv"
SPECIES_NAME = "Bradyrhizobium"
```

Then run:
```python
python Evaluation/GC & Kmer.py
```

The script will produce two publication‑ready PDF figures:
- gc_content_comparison_boxplot_<species>.pdf
- k3_mer_frequency_comparison_<species>.pdf

### 3. Motif Analysis (motif_analysis.py)
**Usage** – Adjust the BASE_DIR and SPECIES_DIRS variables to point to your generated sequence CSV files. The script expects each species subfolder to contain a prediction_result.csv file with at least seq and label columns.

```python
python Evaluation/motif_analysis.py
```

Outputs:
- A combined figure motif_top5_per_species_beautiful.png / .pdf showing top‑5 enriched 6‑mers for each species.
- A console table with counts of known core promoter motifs.

**Note:** The scripts are pre‑configured to work with the folder structure of this repository. If you use your own data, simply update the file paths accordingly.
