#%%

import pandas as pd
import matplotlib.pyplot as plt

raw_df = pd.read_csv('raw.tsv', sep='\t', usecols=range(6))
raw_df = raw_df[raw_df['TAGS'].str.startswith('genre---')]

raw_df['genre'] = raw_df['TAGS'].str[8:]

# %%
value_counts = raw_df['genre'].value_counts()

value_counts.plot(kind='bar')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('Histogram of Unique Values in Column')
plt.show()

# %%
len(raw_df['TAGS'].unique())

# %%
value_counts
# %%
