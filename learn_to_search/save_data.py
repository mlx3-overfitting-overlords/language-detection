# Description: This script parses the MS MARCO dataset and creates a new dataset with the following columns:
# query_text, query_id, passage_text, passage_url, label
#
# The label is 1 if the passage is relevant to the query, and 0 otherwise.
#
# The script is based on the following tutorial: https://towardsdatascience.com/using-bert-for-query-answering-cb6daa40f0da
#

import pandas as pd
import random
import tqdm
import pyarrow 

def save_data(file_suffix):
    df = pd.read_parquet(f'./data/ms_marco_{file_suffix}.parquet')#.sample(n=20000, random_state=42)
    data_samples = []

    for idx, row in tqdm.tqdm(list(df.iterrows()),desc=f"Parsing data", unit="row"):
        query_text = row['query']
        query_id = row['query_id']
        passages_dict = row['passages']

        positive_passage_texts = passages_dict['passage_text']
        positive_passage_urls = passages_dict['url']
        mask = ~df.index.isin([idx])
        random_passages = [
            str(item) for sublist in df.loc[mask].sample(n=100, random_state=42)['passages'].apply(
                lambda x: x['passage_text']
            ).tolist() for item in sublist
        ]        
        negative_passage_texts = [p for p in random_passages]

        for passage_text, url in zip(positive_passage_texts, positive_passage_urls):
            data_samples.append({
                'query_text': query_text,
                'query_index': row['query_id'],
                'passage_text': passage_text,
                'passage_url': url,
                'label': 1
            })

            negative_passage_text = random.choice(negative_passage_texts)

            
            data_samples.append({
                'query_text': query_text,
                'query_index': query_id,
                'passage_text': negative_passage_text,
                'passage_url': None,
                'label': 0
            })

    # Convert list of dictionaries to Pandas DataFrame
    df_to_save = pd.DataFrame(data_samples)

    # Convert DataFrame to Parquet Table
    table = pyarrow.Table.from_pandas(df_to_save)

    # Write to a Parquet file
    pyarrow.parquet.write_table(table, './data/data2.parquet')

save_data('train')
