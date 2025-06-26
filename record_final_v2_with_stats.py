import pandas as pd
import re
from rapidfuzz import fuzz, process
from collections import defaultdict

# -----------------------
# Step 1: 数据加载与清洗
# -----------------------

def clean_name(name):
    if pd.isna(name):
        return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def ultraclean(name):
    if pd.isna(name):
        return ""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def get_char_ngrams(s, n=3):
    return set(s[i:i+n] for i in range(len(s)-n+1)) if len(s) >= n else set()

def jaccard_similarity(a, b, n=3):
    ngrams_a = get_char_ngrams(a, n)
    ngrams_b = get_char_ngrams(b, n)
    if not ngrams_a or not ngrams_b:
        return 0
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

def hybrid_score(x, y, **kwargs):
    return (
        0.4 * fuzz.token_sort_ratio(x, y) +
        0.3 * fuzz.partial_ratio(x, y) +
        0.3 * fuzz.ratio(x, y)
    )

# -----------------------
# Step 2: 数据准备
# -----------------------

primary = pd.read_csv("primary.csv")
alternate = pd.read_csv("alternate.csv")
# test = pd.read_excel("test_02(1).xlsx", sheet_name="Sheet1")
test = pd.read_csv("test_01.csv")        # 包含: ID, VARIANT

primary['CLEAN_NAME'] = primary['NAME'].apply(clean_name)
alternate['CLEAN_NAME'] = alternate['NAME'].apply(clean_name)
test['CLEAN_NAME'] = test['VARIANT'].apply(clean_name)

primary['ULTRA_NAME'] = primary['NAME'].apply(ultraclean)
alternate['ULTRA_NAME'] = alternate['NAME'].apply(ultraclean)
test['ULTRA_NAME'] = test['VARIANT'].apply(ultraclean)

reference = pd.concat([
    primary[['ID', 'CLEAN_NAME', 'ULTRA_NAME']],
    alternate[['ID', 'CLEAN_NAME', 'ULTRA_NAME']]
], ignore_index=True)

ultra_to_ids = defaultdict(set)
for _, row in reference.iterrows():
    ultra_to_ids[row['ULTRA_NAME']].add(row['ID'])

# -----------------------
# Step 3: 匹配逻辑
# -----------------------

def match_name(row, reference_df, score_threshold=85, fallback_n=3):
    variant_clean = row['CLEAN_NAME']
    variant_ultra = row['ULTRA_NAME']

    # 第一阶段：正常匹配
    choices = reference_df['CLEAN_NAME'].tolist()
    match = process.extractOne(variant_clean, choices, scorer=hybrid_score)
    if match:
        matched_name, score, idx = match
        matched_id = reference_df.iloc[idx]['ID']
        if score >= score_threshold:
            return pd.Series([matched_id, matched_name, score, 'hybrid'])

    # 第二阶段：fallback 走 Jaccard
    best_score = -1
    best_id = None
    best_ultra = None
    for _, row_ref in reference_df.iterrows():
        score = jaccard_similarity(variant_ultra, row_ref['ULTRA_NAME'], n=fallback_n)
        if score > best_score:
            best_score = score
            best_id = row_ref['ID']
            best_ultra = row_ref['ULTRA_NAME']
    return pd.Series([best_id, best_ultra, best_score * 100, 'jaccard'])

# -----------------------
# Step 4: 执行匹配
# -----------------------

test_sampled = test.sample(n=300, random_state=42).reset_index(drop=True)

test_sampled[['MATCHED_ID', 'MATCHED_NAME', 'SCORE', 'STRATEGY']] = test_sampled.apply(
    lambda row: match_name(row, reference), axis=1
)

# -----------------------
# Step 5: 正确性判断
# -----------------------

test_sampled['CORRECT'] = test_sampled.apply(
    lambda row: row['ID'] in ultra_to_ids.get(ultraclean(row['MATCHED_NAME']), set()),
    axis=1
)

# -----------------------
# Step 6: 结果分析
# -----------------------

print(test_sampled[['VARIANT', 'MATCHED_NAME', 'SCORE', 'ID', 'CORRECT', 'STRATEGY']])
print(f"融合策略准确率: {test_sampled['CORRECT'].mean():.2%}")

def compute_classification_metrics(df, score_threshold=85):
    # 条件划分
    df['PREDICTED'] = df['SCORE'] >= score_threshold
    # df['MATCH_CORRECT'] = df.apply(
    #     lambda row: row['ID'] in ultra_to_ids.get(ultraclean(row['MATCHED_NAME']), set()) if row['PREDICTED'] else False,
    #     axis=1
    # )

    df['MATCH_CORRECT'] = df['CORRECT']

    TP = ((df['PREDICTED'] == True) & (df['MATCH_CORRECT'] == True)).sum()
    FP = ((df['PREDICTED'] == True) & (df['MATCH_CORRECT'] == False)).sum()
    FN = ((df['PREDICTED'] == False) & (df['MATCH_CORRECT'] == False)).sum()
    TN = ((df['PREDICTED'] == False) & (df['MATCH_CORRECT'] == True)).sum()  # 如果某些任务中没有真值

    precision = TP / (TP + FP) if TP + FP else 0
    recall    = TP / (TP + FN) if TP + FN else 0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy  = (TP + TN) / len(df)

    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print(f"Accuracy:  {accuracy:.2%}")

compute_classification_metrics(test_sampled, score_threshold=85)
