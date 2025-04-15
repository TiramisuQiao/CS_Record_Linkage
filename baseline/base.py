from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
from rich import print
import re
from ftfy import fix_text
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def ngrams(string, n=3):
    string = fix_text(string) # fix text
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

file_str = '/home/tlmsq/datamining/CS_Record_Linkage/dataset/alternate.csv'
data = pd.read_csv(file_str)

# Print original statistics
print(f"Original number of records: {len(data)}")
print(f"Original number of unique names: {len(data['NAME'].unique())}")
print(f"Original number of unique IDs: {len(data['ID'].unique())}")

# Create ground truth groups based on ID
id_groups = defaultdict(list)
for _, row in data.iterrows():
    id_groups[row['ID']].append(row['NAME'])

# Count companies with multiple names
multiple_names_companies = sum(1 for names in id_groups.values() if len(names) > 1)
print(f"Number of companies with multiple names: {multiple_names_companies}")

# TF-IDF processing
from sklearn.feature_extraction.text import TfidfVectorizer
names = data['NAME'].unique()
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(names)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(tf_idf_matrix)

# Set threshold for similarity
threshold = 0.8

# Find similar names using TF-IDF
similar_names = []
processed_indices = set()

for i in range(len(names)):
    if i in processed_indices:
        continue
        
    similar_indices = np.where(similarity_matrix[i] > threshold)[0]
    if len(similar_indices) > 1:  # If there are similar names
        group = [(names[j], j) for j in similar_indices]
        similar_names.append(group)
        processed_indices.update(similar_indices)

# Create mapping dictionary for name standardization
name_mapping = {}
for group in similar_names:
    # Use the first name in group as standard name
    standard_name = group[0][0]
    for name, _ in group:
        name_mapping[name] = standard_name

# Apply mapping to create standardized dataset
data['STANDARDIZED_NAME'] = data['NAME'].map(lambda x: name_mapping.get(x, x))

# Evaluate accuracy
correct_matches = 0
incorrect_matches = 0
missed_matches = 0

# Check each ID group
for company_id, true_names in id_groups.items():
    if len(true_names) > 1:
        # Get standardized names for this ID
        standardized_names = set(data[data['ID'] == company_id]['STANDARDIZED_NAME'])
        
        # If all names were mapped to the same standardized name
        if len(standardized_names) == 1:
            correct_matches += 1
        else:
            missed_matches += 1

# Check for false positives (different IDs mapped to same standardized name)
standardized_groups = defaultdict(set)
for _, row in data.iterrows():
    standardized_groups[row['STANDARDIZED_NAME']].add(row['ID'])

false_positives = sum(1 for ids in standardized_groups.values() if len(ids) > 1)
incorrect_matches = false_positives

# Calculate metrics
total_companies_with_multiple_names = multiple_names_companies
precision = correct_matches / (correct_matches + incorrect_matches) if (correct_matches + incorrect_matches) > 0 else 0
recall = correct_matches / total_companies_with_multiple_names if total_companies_with_multiple_names > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\nEvaluation Results:")
print(f"Correct matches: {correct_matches}")
print(f"Incorrect matches (false positives): {incorrect_matches}")
print(f"Missed matches: {missed_matches}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1_score:.3f}")

# Print results
print(f"\nAfter processing:")
print(f"Number of unique standardized names: {len(data['STANDARDIZED_NAME'].unique())}")

# Export results with evaluation metrics
output_file = file_str.replace('.csv', '_standardized.csv')
data.to_csv(output_file, index=False)
print(f"\nResults exported to: {output_file}")

# Export detailed matching results
matching_results = []
for company_id, true_names in id_groups.items():
    if len(true_names) > 1:
        standardized_names = set(data[data['ID'] == company_id]['STANDARDIZED_NAME'])
        matching_results.append({
            'ID': company_id,
            'Original_Names': ', '.join(true_names),
            'Standardized_Names': ', '.join(standardized_names),
            'Correctly_Matched': len(standardized_names) == 1
        })

matching_results_df = pd.DataFrame(matching_results)
matching_results_file = file_str.replace('.csv', '_matching_results.csv')
matching_results_df.to_csv(matching_results_file, index=False)
print(f"Detailed matching results exported to: {matching_results_file}")