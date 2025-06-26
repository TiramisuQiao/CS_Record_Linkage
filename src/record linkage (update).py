import pandas as pd
import re
from rapidfuzz import fuzz, process

primary = pd.read_csv("primary.csv")     # 包含: ID, NAME, TYPE
alternate = pd.read_csv("alternate.csv") # 包含: ID, NAME
test = pd.read_csv("test_01.csv")           # 包含: ID, VARIANT


# 加入来源标注
primary['SOURCE'] = 'primary'
alternate['SOURCE'] = 'alternate'

# 数据清洗函数
def clean_name(name):
    if pd.isna(name):
        return ""
    name = name.lower()                        # 小写化
    name = re.sub(r'[^\w\s]', '', name)        # 移除标点符号
    name = re.sub(r'\s+', ' ', name).strip()   # 多空格合一
    return name

# 应用清洗
primary['CLEAN_NAME'] = primary['NAME'].apply(clean_name)
alternate['CLEAN_NAME'] = alternate['NAME'].apply(clean_name)
test['CLEAN_NAME'] = test['VARIANT'].apply(clean_name)

# 合并 primary 和 alternate 成一个 reference 匹配库
reference = pd.concat([
    primary[['ID', 'CLEAN_NAME']],
    alternate[['ID', 'CLEAN_NAME']]
]).drop_duplicates().reset_index(drop=True)

# 显示前几行看看处理效果
# print(reference[reference['ID'] == 36])
# print(reference.head(100))
# print(test.head())

# 匹配函数：从 reference 中找 test 中每一条记录最接近的候选项
def match_variant_to_reference(variant, reference_df):
    choices = reference_df['CLEAN_NAME'].tolist()
    match = process.extractOne(variant, choices, scorer=fuzz.token_sort_ratio)
    if match:
        best_match_name, score, idx = match
        best_match_id = reference_df.iloc[idx]['ID']
        return pd.Series([best_match_id, score])
    else:
        return pd.Series([None, 0])
# 只保留300条随机样本（固定 random_state 可复现）
test_sampled = test.sample(n=300, random_state=42).reset_index(drop=True)

# 对每个 test 名称做匹配
test_sampled[['MATCHED_ID', 'SCORE']] = test_sampled['CLEAN_NAME'].apply(
    lambda x: match_variant_to_reference(x, reference)
)

thres = 80.0

test_sampled['CORRECT'] = test_sampled['ID'] == test_sampled['MATCHED_ID']


# 查看匹配结果（包括真实 ID 和匹配 ID 之间是否一致）
print(test_sampled[['VARIANT', 'MATCHED_ID', 'SCORE', 'ID', 'CORRECT']])
# print(test_sampled[['VARIANT', 'CLEAN_NAME', 'MATCHED_ID', 'SCORE', 'ID', 'CORRECT']].head())
print(f"匹配准确率（300条样本）: {test_sampled['CORRECT'].mean():.2%}")


# 匹配正确
correct = test_sampled['MATCHED_ID'] == test_sampled['ID']
# 置信度通过
confident = test_sampled['SCORE'] >= thres

# 分类四种情况：
TP = ((correct) & (confident)).sum()
FP = ((~correct) & (confident)).sum()
FN = ((~correct) & (~confident)).sum()
TN = ((correct) & (~confident)).sum()  # 可选统计

print(f"✅ True Positives  (正确匹配，score高): {TP}")
print(f"❌ False Positives (错误匹配，score高): {FP}")
print(f"❌ False Negatives (正确匹配，但score低被拒绝): {FN}")
print(f"✅ True Negatives  (错误匹配，score低被拒绝): {TN}")

precision = TP / (TP + FP) if TP + FP > 0 else 0
recall    = TP / (TP + FN) if TP + FN > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
accuracy  = (TP + TN) / len(test_sampled)

print(f"\n🎯 Precision: {precision:.2%}")
print(f"🎯 Recall:    {recall:.2%}")
print(f"🎯 F1 Score:  {f1:.2%}")
print(f"🎯 Accuracy (包含拒绝): {accuracy:.2%}")
'''
import matplotlib.pyplot as plt

plt.hist(test_sampled['SCORE'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('similarity score')
plt.ylabel('sample number')
plt.title('matching score distribution（sampled 300）')
plt.grid(True)
plt.show()
'''




