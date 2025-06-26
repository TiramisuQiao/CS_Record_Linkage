import pandas as pd
import re
from rapidfuzz import fuzz, process
from collections import defaultdict

# 加载数据
primary = pd.read_csv("primary.csv")         # 包含 ID, NAME
alternate = pd.read_csv("alternate.csv")     # 包含 ID, NAME
test = pd.read_excel("test_02(1).xlsx", sheet_name="Sheet1")           # 包含 ID, VARIANT

# 定义清洗函数：极限清洗，移除所有标点、空格，只保留字母和数字
def ultraclean(name):
    if pd.isna(name):
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)  # 只保留字母数字
    return name

primary['ULTRA_NAME'] = primary['NAME'].apply(ultraclean)
alternate['ULTRA_NAME'] = alternate['NAME'].apply(ultraclean)
test['ULTRA_NAME'] = test['NAME'].apply(ultraclean)

reference = pd.concat([
    primary[['ID', 'ULTRA_NAME']],
    alternate[['ID', 'ULTRA_NAME']]
], ignore_index=True)

# 构建 ULTRA_NAME → 多个 ID 的映射
name_to_ids = defaultdict(set)
for _, row in reference.iterrows():
    name_to_ids[row['ULTRA_NAME']].add(row['ID'])

def hybrid_score(x, y, **kwargs):
    return (
        0.4 * fuzz.token_sort_ratio(x, y) +
        0.3 * fuzz.partial_ratio(x, y) +
        0.3 * fuzz.ratio(x, y)
    )

def match_variant_ultra(variant, reference_df):
    choices = reference_df['ULTRA_NAME'].tolist()
    match = process.extractOne(variant, choices, scorer=hybrid_score)
    if match:
        best_match_name, score, idx = match
        return pd.Series([best_match_name, score])
    else:
        return pd.Series([None, 0])

test[['MATCHED_NAME', 'SCORE']] = test['ULTRA_NAME'].apply(
    lambda x: match_variant_ultra(x, reference)
)

test['CORRECT'] = test.apply(
    lambda row: row['ID'] in name_to_ids.get(row['MATCHED_NAME'], set()),
    axis=1
)

print(test[['NAME', 'MATCHED_NAME', 'SCORE', 'ID', 'CORRECT']].head())
print(f"匹配准确率（test_02，共{len(test)}条）: {test['CORRECT'].mean():.2%}")
