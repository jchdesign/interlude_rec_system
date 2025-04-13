#%%
import os
import numpy as np
from extract_features import extract_features, process_multiple_songs
from datetime import datetime

# Load in files
file_names = np.array(os.listdir('/volumes/Seagate/mtg-jamendo-dataset/00'))
files = np.char.add('/volumes/Seagate/mtg-jamendo-dataset/00/', file_names)

# #%% 
# # One File
# extract_features('/volumes/Seagate/mtg-jamendo-dataset/00/433600.mp3')

#%% Run multiple files

for folder in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
    file_names = np.array(os.listdir('/volumes/Seagate/mtg-jamendo-dataset/'+folder))
    files = np.char.add('/volumes/Seagate/mtg-jamendo-dataset/'+folder+'/', file_names)
    features_df = process_multiple_songs(files)

    date = datetime.today().strftime('%Y-%m-%d')
    features_df.to_csv('csv/features_df_'+folder+'_'+date+'.csv', index=False)
# %%
