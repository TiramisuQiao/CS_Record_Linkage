import pandas as pd
import re
from rapidfuzz import fuzz, process
from collections import defaultdict

# -----------------------
# 数据读取与准备
# -----------------------

primary = pd.read_csv("primary.csv")     # 包含: ID, NAME, TYPE
alternate = pd.read_csv("alternate.csv") # 包含: ID, NAME
test = pd.read_csv("test_01.csv")        # 包含: ID, VARIANT

primary['SOURCE'] = 'primary'
alternate['SOURCE'] = 'alternate'

# -----------------------
# 文本清洗函数
# -----------------------
def clean_name(name):
    if pd.isna(name):
        return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)      # 移除标点
    name = re.sub(r'\s+', ' ', name).strip() # 多空格合一
    return name

primary['CLEAN_NAME'] = primary['NAME'].apply(clean_name)
alternate['CLEAN_NAME'] = alternate['NAME'].apply(clean_name)
test['CLEAN_NAME'] = test['VARIANT'].apply(clean_name)
"""
primary['CLEAN_NAME'] = primary['NAME'].apply(ultraclean)
alternate['CLEAN_NAME'] = alternate['NAME'].apply(ultraclean)
test['CLEAN_NAME'] = test['VARIANT'].apply(ultraclean)
"""
# -----------------------
# 构建 reference（不去重 ID，保留全部信息）
# -----------------------

reference = pd.concat([
    primary[['ID', 'CLEAN_NAME']],
    alternate[['ID', 'CLEAN_NAME']]
], ignore_index=True)

# 构建 name → ID 集合映射（支持同名多 ID）
name_to_ids = defaultdict(set)
for _, row in reference.iterrows():
    name_to_ids[row['CLEAN_NAME']].add(row['ID'])

# -----------------------
# 匹配函数
# -----------------------

def hybrid_score(x, y, **kwargs):
    return (
        0.4 * fuzz.token_sort_ratio(x, y) +
        0.3 * fuzz.partial_ratio(x, y) +
        0.3 * fuzz.ratio(x, y)
    )

def match_variant_to_reference(variant, reference_df):
    choices = reference_df['CLEAN_NAME'].tolist()
    match = process.extractOne(variant, choices, scorer=hybrid_score)
    if match:
        best_match_name, score, idx = match
        return pd.Series([best_match_name, score])
    else:
        return pd.Series([None, 0])

# -----------------------
# 样本选择 & 匹配执行
# -----------------------

test_sampled = test.sample(n=300, random_state=42).reset_index(drop=True)

test_sampled[['MATCHED_NAME', 'SCORE']] = test_sampled['CLEAN_NAME'].apply(
    lambda x: match_variant_to_reference(x, reference)
)

# 判断是否正确：只要 test 的 ID 属于该 name 对应的所有 ID 之一
test_sampled['CORRECT'] = test_sampled.apply(
    lambda row: row['ID'] in name_to_ids.get(row['MATCHED_NAME'], set()),
    axis=1
)

# -----------------------
# 评估输出
# -----------------------

print(test_sampled[['VARIANT', 'MATCHED_NAME', 'SCORE', 'ID', 'CORRECT']])
print(f"匹配准确率（3000条样本）: {test_sampled['CORRECT'].mean():.2%}")

# -----------------------
# 错误样本分析导出
# -----------------------

failed = test_sampled[test_sampled['CORRECT'] == False]

# 获取 test ID 对应的“正确 name”（用于 debug）
# id2clean = reference.drop_duplicates('ID').set_index('ID')['CLEAN_NAME'].to_dict()
# test_sampled['TRUE_CLEAN_NAME'] = test_sampled['ID'].map(id2clean)

#failed[['VARIANT', 'CLEAN_NAME', 'MATCHED_NAME', 'ID', 'SCORE', 'TRUE_CLEAN_NAME']].to_csv(
#    'failed_matches.csv', index=False, encoding='utf-8-sig'
# )
