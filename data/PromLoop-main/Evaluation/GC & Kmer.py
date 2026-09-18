import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import itertools

# --- 配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局字体与绘图参数优化
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'savefig.dpi': 1200,
})

# 文件路径
GENERATED_PREDICTIONS_FILE = "../生成部分/Bradyrhizobium/prediction_result.csv"
REAL_SAMPLES_FILE = "../Dataset/Bradyrhizobium/positive_samples.csv"
SPECIES_NAME = "Bradyrhizobium"

GC_OUTPUT_PLOT_PATH = f"../生成部分/Bradyrhizobium/gc_content_comparison_boxplot_{SPECIES_NAME.lower().replace(' ', '_')}.pdf"
KMER_OUTPUT_PLOT_PATH = f"../生成部分/Bradyrhizobium/k3_mer_frequency_comparison_{SPECIES_NAME.lower().replace(' ', '_')}.pdf"


# --- 函数定义 ---
def calculate_gc_content(sequence):
    sequence = sequence.upper()
    g_count, c_count = sequence.count('G'), sequence.count('C')
    total = len(sequence)
    return (g_count + c_count) / total if total > 0 else 0.0


def calculate_kmer_frequencies(sequences, k=3):
    kmer_counts = Counter()
    total_kmers = 0

    for seq in sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if len(kmer) == k:
                kmer_counts[kmer] += 1
                total_kmers += 1

    all_possible = [''.join(p) for p in itertools.product(['A', 'C', 'G', 'T'], repeat=k)]
    freqs = {km: kmer_counts.get(km, 0) / total_kmers if total_kmers > 0 else 0.0 for km in all_possible}
    return dict(sorted(freqs.items()))


# --- 主程序 ---
def main():
    print("Reading data...")
    pred_df = pd.read_csv(GENERATED_PREDICTIONS_FILE)
    generated_seqs = pred_df[pred_df['label'] == 1]['seq'].tolist()

    real_df = pd.read_csv(REAL_SAMPLES_FILE)
    real_seqs = real_df['seq'].tolist()

    print(f"Generated positive: {len(generated_seqs)} | Real positive: {len(real_seqs)}")

    # === 1. GC Content Boxplot ===
    gc_gen = [calculate_gc_content(s) for s in generated_seqs]
    gc_real = [calculate_gc_content(s) for s in real_seqs]

    q1_g, med_g, q3_g = np.percentile(gc_gen, [25, 50, 75])
    q1_r, med_r, q3_r = np.percentile(gc_real, [25, 50, 75])

    fig, ax = plt.subplots(figsize=(8, 6))

    boxprops = dict(linewidth=1.2, facecolor='#a8dadc', edgecolor='black')
    medianprops = dict(linewidth=1.8, color='darkslategray')
    whiskerprops = dict(linewidth=1.0, color='gray')
    capprops = dict(linewidth=1.0, color='gray')

    bp = ax.boxplot([gc_gen, gc_real],
                    labels=['Generated', 'Real'],
                    patch_artist=True,
                    boxprops=boxprops,
                    medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    flierprops=dict(marker='o', markersize=4, alpha=0.7))

    # 添加统计标注（位置微调以避免重叠）
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset_y = y_range * 0.012  # 动态偏移量

    # Generated
    ax.axhline(y=med_g, color='tab:blue', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.text(ax.get_xlim()[0] - 0.38, med_g, f'Med: {med_g:.3f}',
            ha='right', va='center', fontsize=9, fontweight='bold', color='tab:blue')
    ax.axhline(y=q3_g, color='tab:blue', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[0] - 0.38, q3_g, f'Q3: {q3_g:.3f}',
            ha='right', va='center', fontsize=9, color='tab:blue')
    ax.axhline(y=q1_g, color='tab:blue', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[0] - 0.38, q1_g, f'Q1: {q1_g:.3f}',
            ha='right', va='center', fontsize=9, color='tab:blue')

    # Real
    ax.axhline(y=med_r, color='tab:orange', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.text(ax.get_xlim()[1] + 0.38, med_r, f'Med: {med_r:.3f}',
            ha='left', va='center', fontsize=9, fontweight='bold', color='tab:orange')
    ax.axhline(y=q3_r, color='tab:orange', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[1] + 0.38, q3_r, f'Q3: {q3_r:.3f}',
            ha='left', va='center', fontsize=9, color='tab:orange')
    ax.axhline(y=q1_r, color='tab:orange', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[1] + 0.38, q1_r, f'Q1: {q1_r:.3f}',
            ha='left', va='center', fontsize=9, color='tab:orange')

    ax.set_title(f'GC Content Distribution Comparison ({SPECIES_NAME})\nGenerated vs. Real Positive Samples',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('GC Content', fontsize=12)
    ax.set_xlabel('Sample Source', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(pad=1.0)  # 留出足够边距，避免裁剪
    plt.savefig(GC_OUTPUT_PLOT_PATH,
                bbox_inches='tight',
                pad_inches=0.1,
                format='pdf')  # 生成 1200 dpi PDF
    plt.show()
    print(f"[✓] GC plot saved to: {GC_OUTPUT_PLOT_PATH} (PDF, 1200 dpi)")

    # === 2. 3-mer Frequency Bar Plot ===
    k = 3
    print(f"Calculating {k}-mer frequencies...")
    freq_gen = calculate_kmer_frequencies(generated_seqs, k)
    freq_real = calculate_kmer_frequencies(real_seqs, k)

    kmers = list(freq_gen.keys())
    vals_gen, vals_real = list(freq_gen.values()), list(freq_real.values())

    fig, ax = plt.subplots(figsize=(24, 8))  # 宽度24英寸，容纳64个标签
    bars1 = ax.bar(x=np.arange(len(kmers)) - 0.175,
                   height=vals_gen,
                   width=0.35,
                   label='Generated',
                   color='#4e79a7',
                   alpha=0.85)
    bars2 = ax.bar(x=np.arange(len(kmers)) + 0.175,
                   height=vals_real,
                   width=0.35,
                   label='Real',
                   color='#f28e2c',
                   alpha=0.85)

    ax.set_xlabel(f'{k}-mer Type', fontsize=12)
    ax.set_ylabel(f'{k}-mer Frequency', fontsize=12)
    ax.set_title(f'{k}-mer Frequency Distribution ({SPECIES_NAME})\nGenerated vs. Real Positive Samples',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(kmers)))
    ax.set_xticklabels(kmers, rotation=60, ha='right', fontsize=9)  # 9pt 保证64个标签清晰
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(pad=1.2)  # 更大边距，防止标签被切
    plt.savefig(KMER_OUTPUT_PLOT_PATH,
                bbox_inches='tight',
                pad_inches=0.1,
                format='pdf')  # 生成 1200 dpi PDF
    plt.show()
    print(f"[✓] {k}-mer plot saved to: {KMER_OUTPUT_PLOT_PATH} (PDF, 1200 dpi)")


if __name__ == "__main__":
    main()