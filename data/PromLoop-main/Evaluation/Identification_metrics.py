import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)


def calculate_classification_metrics(true_labels, pred_labels, pred_probs=None):
    """
    计算二分类任务的评估指标

    参数:
    true_labels: 真实标签列表/数组
    pred_labels: 预测标签列表/数组
    pred_probs: 预测为正类的概率列表/数组（用于计算AUC），可选

    返回:
    dict: 包含所有指标结果的字典
    """

    # 确保输入格式正确
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    if len(true_labels) != len(pred_labels):
        raise ValueError("真实标签和预测标签数量不一致")

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels, labels=[0, 1]).ravel()

    # 计算各项指标
    metrics = {}

    # 1. 准确率
    metrics['accuracy'] = accuracy_score(true_labels, pred_labels)

    # 2. 精确率
    # 注意：当没有正类预测时，sklearn默认返回0，可以通过zero_division参数控制
    metrics['precision'] = precision_score(true_labels, pred_labels, zero_division=0)

    # 3. 召回率
    metrics['recall'] = recall_score(true_labels, pred_labels, zero_division=0)

    # 4. F1分数
    metrics['f1_score'] = f1_score(true_labels, pred_labels, zero_division=0)

    # 5. MCC
    metrics['mcc'] = matthews_corrcoef(true_labels, pred_labels)

    # 6. AUC
    # 如果有预测概率，计算AUC
    if pred_probs is not None:
        pred_probs = np.array(pred_probs)
        if len(pred_probs) != len(true_labels):
            raise ValueError("预测概率与标签数量不一致")
        try:
            metrics['auc'] = roc_auc_score(true_labels, pred_probs)
        except ValueError as e:
            print(f"警告: 无法计算AUC: {e}")
            print("这可能是因为数据中只有一个类别，将AUC设置为NaN")
            metrics['auc'] = np.nan
    else:
        print("注意: 未提供预测概率，无法计算AUC")
        metrics['auc'] = np.nan

    # 7. 额外添加：混淆矩阵元素（便于查看）
    metrics['confusion_matrix'] = {
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp)
    }

    # 8. 额外添加：支持度（正类样本数量）
    metrics['support'] = int(np.sum(true_labels == 1))

    return metrics


def load_and_calculate(true_file, pred_file,
                       true_label_col='label',
                       pred_label_col='label',
                       pred_prob_col=None):
    """
    从CSV文件加载数据并计算指标

    参数:
    true_file: 包含真实标签的CSV文件路径
    pred_file: 包含预测标签的CSV文件路径
    true_label_col: 真实标签列名
    pred_label_col: 预测标签列名
    pred_prob_col: 预测概率列名（可选，用于计算AUC）
    """

    # 加载数据
    true_df = pd.read_csv(true_file)
    pred_df = pd.read_csv(pred_file)

    print(f"真实标签文件行数: {len(true_df)}")
    print(f"预测标签文件行数: {len(pred_df)}")

    # 提取标签
    true_labels = true_df[true_label_col].values
    pred_labels = pred_df[pred_label_col].values

    # 提取预测概率（如果提供）
    pred_probs = None
    if pred_prob_col is not None and pred_prob_col in pred_df.columns:
        pred_probs = pred_df[pred_prob_col].values
        print(f"使用 '{pred_prob_col}' 列作为预测概率计算AUC")
    elif pred_prob_col is not None:
        print(f"警告: 未找到列 '{pred_prob_col}'，将不使用预测概率")

    # 计算指标
    metrics = calculate_classification_metrics(true_labels, pred_labels, pred_probs)

    return metrics, true_labels, pred_labels


def print_metrics(metrics):
    """格式化打印指标结果"""

    print("\n" + "=" * 60)
    print("二分类评估指标结果")
    print("=" * 60)

    # 打印混淆矩阵
    cm = metrics['confusion_matrix']
    print(f"\n混淆矩阵:")
    print(f"          预测负类(0)   预测正类(1)")
    print(f"实际负类(0)   {cm['true_negative']:>6}         {cm['false_positive']:>6}")
    print(f"实际正类(1)   {cm['false_negative']:>6}         {cm['true_positive']:>6}")

    print(f"\n正类样本数（支持度）: {metrics['support']}")

    print(f"\n主要指标:")
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score):  {metrics['f1_score']:.4f}")
    print(f"马修斯系数 (MCC):   {metrics['mcc']:.4f}")

    if not np.isnan(metrics['auc']):
        print(f"AUC:               {metrics['auc']:.4f}")
    else:
        print("AUC:               未计算")

    print("=" * 60)


# 实际使用示例
if __name__ == "__main__":
    # 如果您的文件如下：
    # true_labels.csv: 包含真实标签，列名为 'label'
    # predictions.csv: 包含预测标签，列名为 'pred_label'，预测概率列名为 'probability'

    metrics, true_labels, pred_labels = load_and_calculate(
        '../Datasets/Bradyrhizobium/test.csv',  # 真实标签文件路径
        '../Dataset/results/lucaone-gene_luca_base/Bradyrhizobium_prediction_results.csv',  # 预测标签文件路径
        true_label_col='label',  # 真实标签列名
        pred_label_col='label',  # 预测标签列名
        pred_prob_col='prob'  # 预测概率列名（用于计算AUC）
    )

    print_metrics(metrics)
