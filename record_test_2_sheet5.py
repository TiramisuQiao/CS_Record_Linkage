import pandas as pd
import re
from collections import defaultdict

# -------------------------
# 1. 加载数据
# -------------------------
primary = pd.read_csv("primary.csv")
alternate = pd.read_csv("alternate.csv")
test = pd.read_excel("test_02(1).xlsx", sheet_name="Sheet5")  # 注意换 sheet

# -------------------------
# 2. Token 清洗函数（word-level）
# -------------------------
def tokenize_clean(s):
    if pd.isna(s):
        return []
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)  # 去掉标点
    return s.strip().split()

# -------------------------
# 3. 应用清洗
# -------------------------
primary['TOKENS'] = primary['NAME'].apply(tokenize_clean)
alternate['TOKENS'] = alternate['NAME'].apply(tokenize_clean)
test['TOKENS'] = test['NAME'].apply(tokenize_clean)

# 构造参考库
reference = pd.concat([primary[['ID', 'TOKENS']], alternate[['ID', 'TOKENS']]], ignore_index=True)

# 用于容错（多个ID可能对应同一tokens）
tokens_to_ids = defaultdict(set)
for _, row in reference.iterrows():
    tokens_to_ids[tuple(row['TOKENS'])].add(row['ID'])

# -------------------------
# 4. Token Containment 匹配
# -------------------------
def containment_score(query_tokens, ref_tokens):
    set_q = set(query_tokens)
    set_r = set(ref_tokens)
    if not set_q:
        return 0
    return len(set_q & set_r) / len(set_q)

def match_by_containment(query_tokens, reference_df):
    best_score = -1
    best_tokens = None
    best_id = None
    for _, row in reference_df.iterrows():
        score = containment_score(query_tokens, row['TOKENS'])
        if score > best_score:
            best_score = score
            best_tokens = row['TOKENS']
            best_id = row['ID']
    return pd.Series([best_id, best_tokens, best_score])

# -------------------------
# 5. 执行匹配
# -------------------------
test[['MATCHED_ID', 'MATCHED_TOKENS', 'SCORE']] = test['TOKENS'].apply(
    lambda x: match_by_containment(x, reference)
)

# -------------------------
# 6. 判断是否正确
# -------------------------
test['CORRECT'] = test.apply(
    lambda row: row['ID'] in tokens_to_ids.get(tuple(row['MATCHED_TOKENS']), set()),
    axis=1
)

# -------------------------
# 7. 输出结果
# -------------------------
print(test[['NAME', 'MATCHED_TOKENS', 'SCORE', 'ID', 'CORRECT']].head())
print(f"匹配准确率（Token Containment Ratio）: {test['CORRECT'].mean():.2%}")
