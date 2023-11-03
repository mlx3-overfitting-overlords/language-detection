# Purpose: Extract sentences from MS MARCO dataset and save them to a text file
#

import pandas
import tqdm

def get_corpus(file_suffix):
    df = pandas.read_parquet(f'./data/ms_marco_{file_suffix}.parquet')#.sample(n=10000,random_state=42)
    
    # Extract sentences from 'passage_text' key of the 'passages' column
    passage_sentences = [sentence for passage_list in df['passages'].apply(lambda x: x['passage_text']) for sentence in passage_list]

    # Extract queries as sentences
    query_sentences = df['query'].dropna().tolist()

    # Combine both lists
    sentences = passage_sentences + query_sentences
    
    return sentences
    
    
corpus = get_corpus('train')

# Save the combined list to a text file
with open('corpus.txt', 'w', encoding='utf-8') as f:
    for sentence in tqdm.tqdm(corpus, desc=f"Exporting corpus.txt", unit="batch"):
        f.write(sentence + '\n')
