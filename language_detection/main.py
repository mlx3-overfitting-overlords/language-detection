import pandas as pd
import pyarrow as pa
# Load Parquet data
DATA_FOLDER = './data/'
PATHS = [DATA_FOLDER + 'CL_it-en.parquet', DATA_FOLDER + 'Flores7Lang.parquet']
data = pd.read_parquet(PATHS[0])
# Now, you can use the `data` DataFrame to analyse and manipulate the data.
print(data.head())
