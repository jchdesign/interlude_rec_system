#%%
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from extract_features import extract_features

import glob
import os

# # Get a list of all CSV files in the directory
# csv_files = glob.glob(os.path.join('csv/', '*.csv'))

# # Create an empty list to store DataFrames
# all_df = []

# # Loop through the list of CSV files
# for csv_file in csv_files:
#     # Read each CSV file into a DataFrame
#     df = pd.read_csv(csv_file)
#     # Append the DataFrame to the list
#     all_df.append(df)

# # Concatenate all DataFrames in the list into one DataFrame
# songs_df = pd.concat(all_df, ignore_index=True)

songs_df = pd.read_csv('./csv/merged_with_genre.csv')

songs_df_processed = songs_df.drop(['filename', 'duration'], axis=1)

#%% Normalize features, add weights
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: One-hot encode genre
genre_encoded = pd.get_dummies(songs_df_processed['genre'], prefix='genre')

# Step 2: Drop original genre column and concatenate encoded genres
songs_features = songs_df_processed.drop(columns=['genre'])
songs_features = pd.concat([songs_features, genre_encoded], axis=1)

# Step 3: Standardize all features
scaler = StandardScaler().set_output(transform="pandas")
X_scaled = scaler.fit_transform(songs_features)

# Step 4: Apply weights to numerical features
feature_weights = {
    'tempo': 0.5,               # slight influence, but not dominant
    'acousticness': 0.5,         # minor
    'dynamic_range': 0.2,        # very minor
    'energy': 2.0,               # STRONG
    'brightness': 0.5,           # minor
    'fullness': 0.2,             # very minor
    'density': 0.2,              # very minor
    'instrumentalness': 1.0,     # small
    'danceability': 1.5,         # STRONG
    'valence': 2.0,              # VERY STRONG
    'tension': 1.5,              # STRONG
}

# Step 5: Apply weights to all relevant columns
for col in X_scaled.columns:
    if col in feature_weights:
        X_scaled[col] *= feature_weights[col]
    elif col.startswith("genre_"):
        X_scaled[col] *= 4.0  # or another weight for genre impact

# Step 6: Fit Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
nn_model.fit(X_scaled)


# %%
def recommend_song(features, songs_df, nn_model, scaler, feature_weights):
    test_df = pd.DataFrame([features])

    # --- Handle genre one-hot encoding ---
    # Identify genre columns from the trained data
    genre_columns = [col for col in scaler.feature_names_in_ if col.startswith("genre_")]

    # One-hot encode the input song's genre
    genre_encoded = pd.get_dummies(test_df['genre'], prefix='genre')

    # Add missing genre columns as zeros
    for col in genre_columns:
        if col not in genre_encoded.columns:
            genre_encoded[col] = 0

    # Keep only the relevant genre columns in the right order
    genre_encoded = genre_encoded[genre_columns]

    # Drop original genre column and combine with encoded genre
    test_df = test_df.drop(columns=['genre'])
    test_df = pd.concat([test_df, genre_encoded], axis=1)

    # Keep only columns that the scaler was fit on
    test_df = test_df[scaler.feature_names_in_]

    # --- Scale using pre-fitted scaler ---
    test_scaled = scaler.transform(test_df)

    # --- Apply weights ---
    for col in test_scaled.columns:
        if col in feature_weights:
            test_scaled[col] *= feature_weights[col]
        elif col.startswith("genre_"):
            test_scaled[col] *= 2.0  # or your preferred genre weight

    # --- Find nearest neighbors ---
    distances, indices = nn_model.kneighbors(test_scaled)

    recommended = songs_df.iloc[indices[0][1:]].iloc[0]
    return recommended[['filename']]

# %%
import os
import pandas as pd
from datetime import datetime

for name in ['ankur', 'tj_bro', 'tj_cousin']:
    file_path = os.path.join(name, 'songs')
    features = []

    for song in os.listdir(file_path):
        song_path = os.path.join(file_path, song)

        # Skip non-files and non-audio files
        if not os.path.isfile(song_path) or not song.lower().endswith(('.mp3', '.wav')):
            continue

        try:
            song_features = extract_features(song_path)
            song_features['filename'] = song  # Optional: include filename
            features.append(song_features)
        except Exception as e:
            print(f"❌ Error processing {song}: {e}")

    if features:
        features_df = pd.DataFrame(features)
        date = datetime.today().strftime('%Y-%m-%d')

        # Ensure the recs directory exists
        recs_dir = os.path.join(name, 'recs')
        os.makedirs(recs_dir, exist_ok=True)

        features_df.to_csv(os.path.join(recs_dir, f'features_df_{date}.csv'), index=False)
        print(f"✅ Saved {len(features)} songs for {name}")
    else:
        print(f"⚠️ No features extracted for {name}")

# %%
import os
import pandas as pd
from datetime import datetime

for name in ['ankur', 'tj_bro', 'tj_cousin']:
    file_path = os.path.join(name, 'recs')
    song_features_df = pd.read_csv(os.path.join(file_path, 'features_df_2025-04-23.csv'))

    recs = []

    for _, row in song_features_df.iterrows():
        # Convert each row into a dictionary of features
        input_features = row.to_dict()

        try:
            rec = recommend_song(input_features, songs_df, nn_model, scaler, feature_weights)
            rec['source_song'] = row['filename']  # Keep track of which song this rec came from
            recs.append(rec)
        except Exception as e:
            print(f"❌ Error recommending for {row['filename']}: {e}")

    if recs:
        recs_df = pd.DataFrame(recs)
        date = datetime.today().strftime('%Y-%m-%d')
        recs_df.to_csv(os.path.join(file_path, f'recs_df_{date}.csv'), index=False)
        print(f"✅ Saved recommendations for {name}")
    else:
        print(f"⚠️ No recommendations generated for {name}")


# %%
song_extracted = extract_features('./tj/songs/drake_fps.mp3')
song_extracted['genre'] = 'hiphop'
rec = recommend_song(song_extracted, songs_df, nn_model, scaler, feature_weights)

# %%
