import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# --- 配置 ---
BASE_DIR = "../生成部分"

SPECIES_DIRS = [
    "Bacillus_subtilis",
    "Baumanii",
    "Bradyrhizobium",
    "Diphtheria",
    "Ecoli",
    "Staphylococcus"
]

KNOWN_MOTIFS = {
    "TATA_box": "TATAAA",
    "TATA_var1": "TATATA",
    "DPE": "GCTT",
    "GC_box": "GGGCGG"
}


# --- 函数定义 ---
def load_positive_sequences(species_dir):
    csv_path = os.path.join(BASE_DIR, species_dir, "prediction_result.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️  文件不存在: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns or 'seq' not in df.columns:
        print(f"⚠️  CSV 缺少 'label' 或 'seq' 列: {csv_path}")
        return []
    pos_seqs = df[df['label'] == 1]['seq'].str.upper().tolist()
    print(f"✅ {species_dir}: 加载 {len(pos_seqs)} 条高置信度生成序列")
    return pos_seqs


def extract_kmers(sequences, k=6):
    kmers = []
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if len(kmer) == k and all(c in 'ACGT' for c in kmer):
                kmers.append(kmer)
    return Counter(kmers)


def scan_known_motifs(sequences, motifs_dict):
    counts = {}
    for name, pattern in motifs_dict.items():
        total = sum(seq.count(pattern.upper()) for seq in sequences)
        counts[name] = total
    return counts


# --- 主程序 ---
def main():
    all_results = {}  # ✅ 现在 all_results 在 main() 内部定义

    for species in SPECIES_DIRS:
        print(f"\n{'=' * 60}\n处理物种: {species}")
        seqs = load_positive_sequences(species)
        if not seqs:
            all_results[species] = {'top_kmers': [], 'known_motifs': {}}
            continue

        kmer_counter = extract_kmers(seqs, k=6)
        top_10 = kmer_counter.most_common(10)
        print(f"Top 10 6-mers:")
        for motif, cnt in top_10:
            print(f"  {motif}: {cnt}")

        known_counts = scan_known_motifs(seqs, KNOWN_MOTIFS)
        print("已知核心 motif 计数:")
        for name, cnt in known_counts.items():
            print(f"  {name}: {cnt}")

        all_results[species] = {
            'top_kmers': top_10,
            'known_motifs': known_counts
        }

    # === 绘图：2x3 网格，竖直柱状图，顶刊级美观风格 ===
    print("\n\n" + "=" * 80)
    print("🎨 绘制：各物种 Top 5 6-mer 频率（竖直柱状图，顶刊配色）")

    # 配色方案
    COLORS = plt.cm.tab10.colors[:6]
    SPECIES_LABELS = [
        r"Bacillus subtilis",
        r"Baumannii",
        r"Bradyrhizobium",
        r"Diphtheria",
        r"Escherichia coli",
        r"Staphylococcus"
    ]

    # 创建 2x3 子图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for idx, (species, ax) in enumerate(zip(SPECIES_DIRS, axes)):
        res = all_results[species]
        top5 = res['top_kmers'][:5]
        motifs = [m for m, _ in top5]
        counts = [c for _, c in top5]

        bars = ax.bar(motifs, counts, color=COLORS[idx], alpha=0.9, edgecolor='black', linewidth=0.5)

        for bar, cnt in zip(bars, counts):
            if cnt > max(counts) * 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                        str(cnt), ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(SPECIES_LABELS[idx], fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)
        ax.set_ylim(0, max(counts) * 1.15)
        ax.set_xlabel('6-mer Motif', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(
        'Top 5 Enriched 6-mers in High-Confidence Generated Promoter Sequences\n'
        '(across six industrial microbial species)',
        fontsize=16, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # 保存高清图
    output_png = os.path.join(BASE_DIR, "motif_top5_per_species_beautiful.png")
    output_pdf = os.path.join(BASE_DIR, "motif_top5_per_species_beautiful.pdf")

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.show()

    print(f"\n✅ 高清图已保存至:\n   PNG: {output_png}\n   PDF: {output_pdf}")

    # --- 汇总统计表 ---
    print("\n" + "=" * 80)
    print("📋 各物种已知核心 motif 总结:")
    header = f"{'Species':<15}"
    for name in KNOWN_MOTIFS.keys():
        header += f"{name:<12}"
    print(header)
    for species, res in all_results.items():
        line = f"{species:<15}"
        for name, cnt in res['known_motifs'].items():
            line += f"{cnt:<12}"
        print(line)


if __name__ == "__main__":
    main()