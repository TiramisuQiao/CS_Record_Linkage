import pandas as pd
import numpy as np
import recordlinkage
from recordlinkage.preprocessing import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import defaultdict
from rich import print
import re
from ftfy import fix_text

def preprocess_text(text):
    """Clean and standardize text data"""
    text = fix_text(text)
    text = text.encode("ascii", errors="ignore").decode()
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_first_char(x):
    """Get first character for blocking"""
    return x[0] if x else ''

# Load data
file_str = '/home/tlmsq/datamining/CS_Record_Linkage/dataset/alternate.csv'
data = pd.read_csv(file_str)

# Print original statistics
print("Original Data Statistics:")
print(f"Total records: {len(data)}")
print(f"Unique names: {len(data['NAME'].unique())}")
print(f"Unique IDs: {len(data['ID'].unique())}")

# Create ground truth groups based on ID
id_groups = defaultdict(list)
for _, row in data.iterrows():
    id_groups[row['ID']].append(row['NAME'])

multiple_names_companies = sum(1 for names in id_groups.values() if len(names) > 1)
print(f"Companies with multiple names: {multiple_names_companies}\n")

# Preprocess names
data['cleaned_name'] = data['NAME'].apply(preprocess_text)
data['blocking_key'] = data['cleaned_name'].apply(get_first_char)

# Method 1: RecordLinkage with blocking
print("Method 1: RecordLinkage with Blocking")
print("-" * 50)

# Create indexer with blocking
indexer = recordlinkage.Index()
indexer.block(left_on='blocking_key', right_on='blocking_key')

# Build pairs with blocking
candidate_pairs = indexer.index(data)
print(f"Number of candidate pairs: {len(candidate_pairs)}")

# Create comparison object
compare = recordlinkage.Compare()

# Add comparison methods
compare.string('cleaned_name', 'cleaned_name', method='jarowinkler', threshold=0.85)
compare.string('cleaned_name', 'cleaned_name', method='levenshtein', threshold=0.85)

# Compute feature vectors
features = compare.compute(candidate_pairs, data)

# Get matches above threshold (any feature match)
matches_recordlinkage = features[features.sum(axis=1) >= 1].index

# Create clusters from matches using connected components
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Create sparse adjacency matrix
n = len(data)
rows, cols = zip(*matches_recordlinkage)
adj_matrix = csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))

# Find connected components
n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
print(f"Number of components found: {n_components}")

# Method 2: TF-IDF + DBSCAN Clustering
print("\nMethod 2: TF-IDF + DBSCAN Clustering")
print("-" * 50)

# Create TF-IDF vectors with reduced features
vectorizer = TfidfVectorizer(
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_features=1000,  # Limit number of features
    analyzer='char',
    ngram_range=(2, 3),
    lowercase=True
)
tfidf_matrix = vectorizer.fit_transform(data['cleaned_name'])

# Perform DBSCAN clustering with optimized parameters
dbscan = DBSCAN(
    eps=0.3,
    min_samples=2,
    metric='cosine',
    n_jobs=-1  # Use all available cores
)
clusters_dbscan = dbscan.fit_predict(tfidf_matrix)

def evaluate_clustering(cluster_labels, data, id_groups):
    """Evaluate clustering results against ground truth"""
    # Create clustering result groups
    cluster_groups = defaultdict(set)
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id != -1:  # Ignore noise points
            name = data.iloc[idx]['NAME']
            true_id = data.iloc[idx]['ID']
            cluster_groups[cluster_id].add((name, true_id))
    
    correct_matches = 0
    incorrect_matches = 0
    missed_matches = 0
    
    # Check each cluster
    for cluster in cluster_groups.values():
        # Get all IDs in this cluster
        cluster_ids = {id_ for _, id_ in cluster}
        
        if len(cluster_ids) == 1:
            # All names in cluster belong to same ID
            correct_matches += 1
        else:
            # Names from different IDs were clustered together
            incorrect_matches += 1
    
    # Check for missed matches
    for true_id, true_names in id_groups.items():
        if len(true_names) > 1:
            # Find all clusters containing these names
            name_clusters = set()
            for name in true_names:
                idx = data[data['NAME'] == name].index[0]
                name_clusters.add(cluster_labels[idx])
            
            if len(name_clusters) > 1:
                missed_matches += 1
    
    # Calculate metrics
    precision = correct_matches / (correct_matches + incorrect_matches) if (correct_matches + incorrect_matches) > 0 else 0
    recall = correct_matches / multiple_names_companies if multiple_names_companies > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct_matches': correct_matches,
        'incorrect_matches': incorrect_matches,
        'missed_matches': missed_matches
    }

# Evaluate both methods
results_rl = evaluate_clustering(labels, data, id_groups)
results_dbscan = evaluate_clustering(clusters_dbscan, data, id_groups)

# Print results
print("\nResults Summary")
print("=" * 50)

print("\nRecordLinkage Results:")
print(f"Precision: {results_rl['precision']:.3f}")
print(f"Recall: {results_rl['recall']:.3f}")
print(f"F1 Score: {results_rl['f1']:.3f}")
print(f"Correct matches: {results_rl['correct_matches']}")
print(f"Incorrect matches: {results_rl['incorrect_matches']}")
print(f"Missed matches: {results_rl['missed_matches']}")

print("\nDBSCAN Clustering Results:")
print(f"Precision: {results_dbscan['precision']:.3f}")
print(f"Recall: {results_dbscan['recall']:.3f}")
print(f"F1 Score: {results_dbscan['f1']:.3f}")
print(f"Correct matches: {results_dbscan['correct_matches']}")
print(f"Incorrect matches: {results_dbscan['incorrect_matches']}")
print(f"Missed matches: {results_dbscan['missed_matches']}")

# Export results
results_df = pd.DataFrame({
    'NAME': data['NAME'],
    'ID': data['ID'],
    'RecordLinkage_Cluster': labels,
    'DBSCAN_Cluster': clusters_dbscan
})

# Add cluster sizes
results_df['RL_Cluster_Size'] = results_df.groupby('RecordLinkage_Cluster')['RecordLinkage_Cluster'].transform('count')
results_df['DBSCAN_Cluster_Size'] = results_df.groupby('DBSCAN_Cluster')['DBSCAN_Cluster'].transform('count')

output_file = file_str.replace('.csv', '_clustering_results.csv')
results_df.to_csv(output_file, index=False)
print(f"\nResults exported to: {output_file}")

# Export detailed analysis
analysis_df = pd.DataFrame({
    'Method': ['RecordLinkage', 'DBSCAN'],
    'Precision': [results_rl['precision'], results_dbscan['precision']],
    'Recall': [results_rl['recall'], results_dbscan['recall']],
    'F1_Score': [results_rl['f1'], results_dbscan['f1']],
    'Correct_Matches': [results_rl['correct_matches'], results_dbscan['correct_matches']],
    'Incorrect_Matches': [results_rl['incorrect_matches'], results_dbscan['incorrect_matches']],
    'Missed_Matches': [results_rl['missed_matches'], results_dbscan['missed_matches']]
})

analysis_file = file_str.replace('.csv', '_analysis.csv')
analysis_df.to_csv(analysis_file, index=False)
print(f"Detailed analysis exported to: {analysis_file}") 