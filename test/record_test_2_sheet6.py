import pandas as pd
import re
from collections import defaultdict

# -------------------------
# 1. 加载数据
# -------------------------
primary = pd.read_csv("primary.csv")
alternate = pd.read_csv("alternate.csv")
test = pd.read_excel("test_02(1).xlsx", sheet_name="Sheet6")  # Sheet6: Word Truncation

# -------------------------
# 2. 分词 + 清洗
# -------------------------
def tokenize(s):
    if pd.isna(s):
        return []
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)  # 去标点
    return s.strip().split()

primary['TOKENS'] = primary['NAME'].apply(tokenize)
alternate['TOKENS'] = alternate['NAME'].apply(tokenize)
test['TOKENS'] = test['NAME'].apply(tokenize)

# -------------------------
# 3. 构建参考集 + 候选词索引
# -------------------------
reference = pd.concat([primary[['ID', 'TOKENS']], alternate[['ID', 'TOKENS']]], ignore_index=True)

# 将每组token序列映射到ID集合（用于判断是否正确）
tokens_to_ids = defaultdict(set)
for _, row in reference.iterrows():
    tokens_to_ids[tuple(row['TOKENS'])].add(row['ID'])

# -------------------------
# 4. token-prefix 匹配得分函数
# -------------------------
def token_prefix_score(query_tokens, ref_tokens):
    if not query_tokens or not ref_tokens:
        return 0
    total = 0
    for qt in query_tokens:
        best = max([1 if rt.startswith(qt) else 0 for rt in ref_tokens], default=0)
        total += best
    return total / len(query_tokens)

# -------------------------
# 5. 匹配函数
# -------------------------
def match_token_prefix(query_tokens, reference_df):
    best_score = -1
    best_id = None
    best_tokens = None
    for _, row in reference_df.iterrows():
        score = token_prefix_score(query_tokens, row['TOKENS'])
        if score > best_score:
            best_score = score
            best_id = row['ID']
            best_tokens = row['TOKENS']
    return pd.Series([best_id, best_tokens, best_score])

# -------------------------
# 6. 执行匹配
# -------------------------
test[['MATCHED_ID', 'MATCHED_TOKENS', 'SCORE']] = test['TOKENS'].apply(
    lambda x: match_token_prefix(x, reference)
)

# -------------------------
# 7. 判断正确性
# -------------------------
test['CORRECT'] = test.apply(
    lambda row: row['ID'] in tokens_to_ids.get(tuple(row['MATCHED_TOKENS']), set()),
    axis=1
)

# -------------------------
# 8. 输出结果
# -------------------------
print(test[['NAME', 'MATCHED_TOKENS', 'SCORE', 'ID', 'CORRECT']].head())
print(f"匹配准确率（Token-prefix, Word Truncation）: {test['CORRECT'].mean():.2%}")
