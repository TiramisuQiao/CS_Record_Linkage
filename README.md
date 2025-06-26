# Bear: Record Linkage Project

A Python project for data cleaning and matching, developed by the team of Junqiao Wang, Zhengtao Zhang, Hengyuan Xu, Chaojia Yu, and Zihao Zhang from Sichuan University-Pittsburgh Institute.

## Project Overview
This project focuses on **record linkage**, aiming to accurately identify and link related records from different data sources. It includes data cleaning, hybrid scoring, and matching algorithms to achieve high precision and recall rates.

## Features
- **Data Cleaning**: Standardize and normalize input data.
- **Matching Algorithm**: Use `rapidfuzz` for efficient fuzzy matching.
- **Hybrid Scoring**: Flexible association of names with multiple IDs.

## Installation
We use uv for package manmagement
```
uv sync
```

## Usage
### Data Cleaning
```python
from src.utils.cleaning import clean_name

cleaned_name = clean_name("Example Name!")
print(cleaned_name)  # Output: "example name"
```

### Matching Records
```python
from src.utils.matching import match_variant_to_reference
import pandas as pd

reference_df = pd.DataFrame({
    'ID': [1, 2],
    'CLEAN_NAME': ["example name", "another name"]
})

result = match_variant_to_reference("exampl name", reference_df)
print(result)  # Output: (1, 95.0)  # ID and similarity score
```

## Dependencies
- Python >= 3.9
- pandas >= 2.3.0
- rapidfuzz >= 3.13.0

## Results
- **Precision**: 98.34%
- **Recall**: 95.70%
- **F1 Score**: 97.00%

## Future Work
- Explore advanced algorithms (e.g., LLMs) for complex data variations.
- Optimize performance using techniques like vectorization and caching.

## Contact
For questions or collaborations, contact:
- Junqiao Wang
- Zhengtao Zhang
- Hengyuan Xu
- Chaojia Yu
- Zihao Zhang