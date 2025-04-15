from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
from rich import print
import re
from ftfy import fix_text

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
# try TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
names = data['NAME'].unique()
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(names)
print( tf_idf_matrix.shape, tf_idf_matrix[5])
# Check if this makes sense: