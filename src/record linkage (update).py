# src/main.py
import pandas as pd
import re
from rapidfuzz import fuzz, process
from utils.cleaning import clean_name
from utils.matching import match_variant_to_reference

def main():
primary = pd.read_csv("primary.csv")     # 包含: ID, NAME, TYPE
alternate = pd.read_csv("alternate.csv") # 包含: ID, NAME
test = pd.read_csv("test_01.csv")           # 包含: ID, VARIANT
    test = pd.read_csv("test_01.csv")        # 包含: ID, VARIANT


# 加入来源标注
primary['SOURCE'] = 'primary'
alternate['SOURCE'] = 'alternate'










# 应用清洗
primary['CLEAN_NAME'] = primary['NAME'].apply(clean_name)
alternate['CLEAN_NAME'] = alternate['NAME'].apply(clean_name)
test['CLEAN_NAME'] = test['VARIANT'].apply(clean_name)

# 合并 primary 和 alternate 成一个 reference 匹配库
reference = pd.concat([
    primary[['ID', 'CLEAN_NAME']],
    alternate[['ID', 'CLEAN_NAME']]
]).drop_duplicates().reset_index(drop=True)
















# 只保留300条随机样本（固定 random_state 可复现）
test_sampled = test.sample(n=300, random_state=42).reset_index(drop=True)

# 对每个 test 名称做匹配
test_sampled[['MATCHED_ID', 'SCORE']] = test_sampled['CLEAN_NAME'].apply(
    lambda x: match_variant_to_reference(x, reference)
)

thres = 80.0

test_sampled['CORRECT'] = test_sampled['ID'] == test_sampled['MATCHED_ID']



    # 查看匹配结果
print(test_sampled[['VARIANT', 'MATCHED_ID', 'SCORE', 'ID', 'CORRECT']])

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





















if __name__ == "__main__":
    main()
