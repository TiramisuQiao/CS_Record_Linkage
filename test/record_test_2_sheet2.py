import pandas as pd
import re
from collections import defaultdict
from rapidfuzz import process, fuzz

# -------------------------
# Step 1: 数据准备
# -------------------------

primary = pd.read_csv("primary.csv")
alternate = pd.read_csv("alternate.csv")
test = pd.read_excel("test_02(1).xlsx", sheet_name="Sheet2")

# -------------------------
# Step 2: 清洗为纯小写压缩串（用于生成 n-gram）
# -------------------------

def ultraclean(s):
    if pd.isna(s):
        return ""
    return re.sub(r'[^a-z0-9]', '', s.lower())

primary['NGRAM_NAME'] = primary['NAME'].apply(ultraclean)
alternate['NGRAM_NAME'] = alternate['NAME'].apply(ultraclean)
test['NGRAM_NAME'] = test['NAME'].apply(ultraclean)

reference = pd.concat([
    primary[['ID', 'NGRAM_NAME']],
    alternate[['ID', 'NGRAM_NAME']]
], ignore_index=True)

# 构建 name → 多个 ID 的映射（防止同名不同 ID）
name_to_ids = defaultdict(set)
for _, row in reference.iterrows():
    name_to_ids[row['NGRAM_NAME']].add(row['ID'])

# -------------------------
# Step 3: 构造 3-gram 集合 + Jaccard 相似度
# -------------------------

def get_char_ngrams(s, n=3):
    return set(s[i:i+n] for i in range(len(s)-n+1)) if len(s) >= n else set()

def jaccard_similarity(a, b, n=3):
    ngrams_a = get_char_ngrams(a, n)
    ngrams_b = get_char_ngrams(b, n)
    if not ngrams_a or not ngrams_b:
        return 0
    return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b)

# reference: 预构建 n-gram 集合列（只做一次）
reference['NGRAM_SET'] = reference['NGRAM_NAME'].apply(lambda x: get_char_ngrams(x, n=3))


# -------------------------
# Step 4: 匹配函数（暴力全比较）
# -------------------------

def jaccard_similarity_cached(query_ngrams, ref_ngrams):
    if not query_ngrams or not ref_ngrams:
        return 0
    return len(query_ngrams & ref_ngrams) / len(query_ngrams | ref_ngrams)

def match_ngram_jaccard_fast_cached(variant, reference_df, n=3):
    query_ngrams = get_char_ngrams(variant, n)
    best_score = -1
    best_id = None
    best_name = None
    for _, row in reference_df.iterrows():
        score = jaccard_similarity_cached(query_ngrams, row['NGRAM_SET'])
        if score > best_score:
            best_score = score
            best_id = row['ID']
            best_name = row['NGRAM_NAME']
    return pd.Series([best_id, best_name, best_score])


# -------------------------
# Step 5: 执行匹配
# -------------------------
test[['MATCHED_ID', 'MATCHED_NAME', 'SCORE']] = test['NGRAM_NAME'].apply(
    lambda x: match_ngram_jaccard_fast_cached(x, reference, n=3)
)

"""
test[['MATCHED_ID', 'MATCHED_NAME', 'SCORE']] = test['NGRAM_NAME'].apply(
    lambda x: match_ngram_jaccard(x, reference, n=3)
)
"""
test['CORRECT'] = test.apply(
    lambda row: row['ID'] in name_to_ids.get(row['MATCHED_NAME'], set()),
    axis=1
)

# -------------------------
# Step 6: 输出与分析
# -------------------------

print(test[['NAME', 'MATCHED_NAME', 'SCORE', 'ID', 'CORRECT']].head())
print(f"匹配准确率（3-gram Jaccard）: {test['CORRECT'].mean():.2%}")


