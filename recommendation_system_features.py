# %%
#%%
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import glob
import os

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join('csv/', '*.csv'))

# Create an empty list to store DataFrames
all_df = []

# Loop through the list of CSV files
for csv_file in csv_files:
    # Read each CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # Append the DataFrame to the list
    all_df.append(df)

# Concatenate all DataFrames in the list into one DataFrame
songs_df = pd.concat(all_df, ignore_index=True)

songs_df_processed = songs_df.drop(['filename', 'duration'], axis=1)

#%% Normalize features, add weights
scaler = StandardScaler()
X_scaled = scaler.fit_transform(songs_df_processed)

# Custom weights
feature_weights = {
    'tempo': 1.5,
    'energy': 1.5,
    'key': 1.0,
    'mode': 1.0,
    'acouticness': 1.0,
    'instrumentalness': 1.0,
    'speechiness': 1.0,
    'danceability': 1.0,
    'valence': 1.0,
    'loudness': 1.0,
    'loudness': 1.0,
}

for col, weight in feature_weights.items():
    if col in X_scaled.columns:
        X_scaled[col] *= weight

# Split
X_train, X_test = train_test_split(X_scaled, train_size=0.8)

# Fit Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=6, algorithm='auto')
nn_model.fit(X_train)

rec_songs = []

for i in range(5):
    query_index = i
    query_vector = X_test[query_index].reshape(1, -1)

    # Get nearest neighbors
    distances, indices = nn_model.kneighbors(query_vector)

    # Show recommended songs (excluding the query song itself)
    recommended = songs_df.iloc[indices[0][1:]].iloc[0]
    # print(songs_df.loc[query_index])
    # print(recommended[['filename']])

    rec_songs.append(recommended[['filename']])

for i in range(5):
    print(rec_songs[i])

# %%
def recommend_song(filepath, songs_df, nn_model):
    scaler = StandardScaler()

    features = extract_features(filepath)
    test_df = pd.DataFrame([features])
    if 'filename' in test_df.columns:
        test_df = test_df.drop(columns='filename')
    test_scaled = scaler.fit_transform(test_df)
    distances, indices = nn_model.kneighbors(test_scaled)
    recommended = songs_df.iloc[indices[0][1:]].iloc[0]
    return recommended[['filename']]

# %%
rec = recommend_song('sidelines.mp3', songs_df, nn_model)
# %%
