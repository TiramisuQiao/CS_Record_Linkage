import pandas as pd
import re
from collections import defaultdict
from rapidfuzz import fuzz, process

# -------------------------
# 1. 读取数据
# -------------------------
primary = pd.read_csv("primary.csv")
alternate = pd.read_csv("alternate.csv")
test = pd.read_excel("test_02(1).xlsx", sheet_name="Sheet7")  # Sheet7 - Initials

# -------------------------
# 2. 获取 token 首字母
# -------------------------
def get_initials(name):
    if pd.isna(name):
        return ""
    name = name.lower()
    tokens = re.split(r'[\s,\.]+', name)
    return ' '.join([t[0] for t in tokens if t])

primary['INITIALS'] = primary['NAME'].apply(get_initials)
alternate['INITIALS'] = alternate['NAME'].apply(get_initials)
test['INITIALS'] = test['NAME'].apply(get_initials)

# 构建 reference 库
reference = pd.concat([
    primary[['ID', 'INITIALS']],
    alternate[['ID', 'INITIALS']]
], ignore_index=True)

# 构建 INITIALS → 多个 ID 的映射
initials_to_ids = defaultdict(set)
for _, row in reference.iterrows():
    initials_to_ids[row['INITIALS']].add(row['ID'])

# -------------------------
# 3. 匹配函数
# -------------------------
def match_by_initials(initial_str, reference_df):
    choices = reference_df['INITIALS'].tolist()
    match = process.extractOne(initial_str, choices, scorer=fuzz.ratio)
    if match:
        best_match, score, idx = match
        return pd.Series([best_match, score])
    else:
        return pd.Series([None, 0])

# -------------------------
# 4. 执行匹配
# -------------------------
test[['MATCHED_INITIALS', 'SCORE']] = test['INITIALS'].apply(
    lambda x: match_by_initials(x, reference)
)

# -------------------------
# 5. 判断正确性
# -------------------------
test['CORRECT'] = test.apply(
    lambda row: row['ID'] in initials_to_ids.get(row['MATCHED_INITIALS'], set()),
    axis=1
)

# -------------------------
# 6. 输出
# -------------------------
print(test[['NAME', 'INITIALS', 'MATCHED_INITIALS', 'SCORE', 'ID', 'CORRECT']].head())
print(f"匹配准确率（基于首字母）: {test['CORRECT'].mean():.2%}")
