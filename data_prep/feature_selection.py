"""
特征选择：覆盖率过滤、方差过滤、共线性去冗余
确保每个临床系统至少保留代表性特征
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MIN_FEATURE_COVERAGE
from config.feature_groups import CLINICAL_GROUPS, feature_to_group

# 每个临床系统需要保留的最少特征数
MIN_PER_GROUP = 2


def select_features(
    df: pd.DataFrame,
    candidate_features: list,
    min_coverage: float = MIN_FEATURE_COVERAGE,
    max_corr: float = 0.95,
    min_variance: float = 0.01,
    max_features: int = 40,
) -> list:
    """
    从候选特征中选择适合因果发现的子集:
    1. 覆盖率 >= min_coverage
    2. 方差 >= min_variance
    3. 高度共线性对中去掉覆盖率更低的
    4. 确保每个临床系统至少保留 MIN_PER_GROUP 个特征
    5. 最多保留 max_features 个
    """
    feats = [f for f in candidate_features if f in df.columns]

    # Step 1: 覆盖率筛选
    coverage = df[feats].notna().mean()
    feats = [f for f in feats if coverage[f] >= min_coverage]
    print(f"    覆盖率 >= {min_coverage}: {len(feats)} 个特征")

    if len(feats) < 3:
        return feats

    # Step 2: 方差筛选
    variances = df[feats].var(skipna=True)
    feats = [f for f in feats if variances[f] >= min_variance]
    print(f"    方差 >= {min_variance}: {len(feats)} 个特征")

    if len(feats) < 3:
        return feats

    # Step 3: 共线性去冗余（仅对 |r| > 0.95 的对）
    corr_matrix = df[feats].corr().abs()
    to_drop = set()
    for i in range(len(feats)):
        if feats[i] in to_drop:
            continue
        for j in range(i + 1, len(feats)):
            if feats[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > max_corr:
                if coverage[feats[i]] >= coverage[feats[j]]:
                    to_drop.add(feats[j])
                else:
                    to_drop.add(feats[i])
    feats = [f for f in feats if f not in to_drop]
    print(f"    去冗余后: {len(feats)} 个特征")

    # Step 4: 如果超过 max_features，确保每个临床系统保留代表后再截取
    if len(feats) > max_features:
        # 先为每个系统预留 top-N 特征
        reserved = set()
        for gname, ginfo in CLINICAL_GROUPS.items():
            group_feats = [f for f in feats if f in ginfo["features"]]
            # 按覆盖率排序，保留 top MIN_PER_GROUP
            group_feats.sort(key=lambda f: coverage[f], reverse=True)
            for f in group_feats[:MIN_PER_GROUP]:
                reserved.add(f)

        # 剩余名额按覆盖率全局排序
        remaining = [f for f in feats if f not in reserved]
        remaining.sort(key=lambda f: coverage[f], reverse=True)
        slots_left = max_features - len(reserved)
        if slots_left > 0:
            reserved.update(remaining[:slots_left])

        feats = [f for f in feats if f in reserved]
        print(f"    系统均衡截取 top {max_features}: {len(feats)} 个特征")

    # 打印各系统分布
    group_counts = {}
    for f in feats:
        g = feature_to_group(f)
        group_counts[g] = group_counts.get(g, 0) + 1
    print(f"    系统分布: {group_counts}")

    return sorted(feats)
