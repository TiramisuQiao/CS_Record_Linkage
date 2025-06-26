import pandas as pd
import re
from rapidfuzz import process, fuzz

# 1. 加载数据
primary = pd.read_csv("primary.csv")
alternate = pd.read_csv("alternate.csv")
test = pd.read_excel("test_02(1).xlsx", sheet_name="Sheet8")

# 2. 统一清洗
def clean(name):
    if pd.isna(name): return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

primary['CLEAN'] = primary['NAME'].apply(clean)
alternate['CLEAN'] = alternate['NAME'].apply(clean)
test['CLEAN'] = test['NAME'].apply(clean)

reference = pd.concat([
    primary[['ID', 'CLEAN']],
    alternate[['ID', 'CLEAN']]
], ignore_index=True)

# 3. 匹配函数（设置阈值）
THRESHOLD = 60

def match_simulated(name, ref_df, threshold=THRESHOLD):
    choices = ref_df['CLEAN'].tolist()
    match = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    if match:
        matched_name, score, idx = match
        matched_id = ref_df.iloc[idx]['ID']
        return pd.Series([matched_name, matched_id, score, False])  # False = 错误匹配
    else:
        return pd.Series([None, None, 0, True])  # True = 正确地没匹配上

# 4. 执行匹配
test[['MATCHED_NAME', 'MATCHED_ID', 'SCORE', 'CORRECT']] = test['CLEAN'].apply(
    lambda x: match_simulated(x, reference)
)

# 5. 评估
print(test[['NAME', 'MATCHED_NAME', 'SCORE', 'CORRECT']].head())
accuracy = test['CORRECT'].mean()
print(f"模拟名字匹配准确率（即未被错误匹配的比例）: {accuracy:.2%}")
