
#%%

import librosa
import librosa.display
import numpy as np
import pandas as pd
from datetime import datetime

def extract_features(audio_path):
    """
    Extract high-level audio features from a music file using Librosa.
    
    Parameters:
    audio_path (str): Path to the audio file
    
    Returns:
    dict: Dictionary of extracted features
    """
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Initialize the feature dictionary
    features = {}
    
    # Duration
    features['duration'] = librosa.get_duration(y=y, sr=sr)
    
    # Tempo (BPM)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    features['tempo'] = tempo
    
    # Energy
    # Energy can be calculated as the RMS (root mean square) energy
    rms = librosa.feature.rms(y=y).mean()
    features['energy'] = float(rms)  # Scale between 0-1 based on your dataset
    
    # Loudness
    # Approximated using the average decibel value
    S = np.abs(librosa.stft(y))
    features['loudness'] = float(librosa.amplitude_to_db(S).mean())
    
    # Key and Mode
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = np.argmax(np.sum(chroma, axis=1))
    features['key'] = int(key)
    
    # Mode (major or minor)
    # This is a simplification - in practice, mode detection is more complex
    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    
    chroma_normalized = np.sum(chroma, axis=1) / np.sum(chroma)
    major_correlation = np.corrcoef(chroma_normalized, major_profile)[0, 1]
    minor_correlation = np.corrcoef(chroma_normalized, minor_profile)[0, 1]
    
    features['mode'] = 1 if major_correlation > minor_correlation else 0  # 1 for major, 0 for minor
    
    # Acousticness
    # Calculated based on spectral contrast and MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Higher spectral contrast often indicates less acoustic sound
    features['acousticness'] = 1.0 - float(np.mean(contrast) / 50.0)  # Scale to 0-1
    
    # Instrumentalness
    # Approximated using the presence of speech-like features
    # Lower MFCC variability in vocals range can indicate more instrumental content
    vocal_mfccs = mfccs[1:5, :]  # Focusing on MFCCs that capture vocal characteristics
    features['instrumentalness'] = 1.0 - min(1.0, float(np.std(vocal_mfccs) / 4.0))
    
    # Speechiness
    # Can be approximated using spectral flatness and energy in speech frequencies
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    
    # Filter signal to focus on speech frequencies (250-3000 Hz)
    y_speech = librosa.effects.preemphasis(y)
    speech_rms = librosa.feature.rms(y=y_speech).mean()
    
    features['speechiness'] = float(min(1.0, (speech_rms / rms) * flatness * 3.0))
    
    # Danceability
    # Approximated using beat strength and regularity
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_strength = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames = librosa.util.fix_frames(beats, x_min=0, x_max=len(beat_strength))
    
    # Calculate beat regularity
    if len(beat_frames) > 1:
        beat_intervals = np.diff(beat_frames)
        beat_regularity = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
    else:
        beat_regularity = 0
    
    valid_frames = beat_frames[beat_frames < len(beat_strength)]
    avg_beat_strength = np.mean(beat_strength[valid_frames]) if len(valid_frames) > 0 else 0
    features['danceability'] = float(min(1.0, 0.5 * beat_regularity + 0.5 * (avg_beat_strength / 0.1)))
    
    # Valence (musical positiveness)
    # Approximated with a combination of mode, tempo, and spectral characteristics
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    mode_factor = 0.5 if features['mode'] == 0 else 0.8  # Minor vs Major
    
    # Higher tempos and spectral centroids often correlate with higher valence
    tempo_normalized = max(0, min(1, (tempo - 50) / 150))  # Normalize tempo between 0-1
    spectral_normalized = max(0, min(1, spectral_centroid / 4000))
    
    features['valence'] = float(mode_factor * 0.4 + tempo_normalized * 0.3 + spectral_normalized * 0.3)
    
    # Liveness
    # Approximated by detecting audience noise and microphone characteristics
    # Variance in higher frequencies can indicate live recordings
    S = np.abs(librosa.stft(y))
    high_freq = S[-int(S.shape[0]/3):, :]  # Focus on high frequency content
    high_freq_var = np.var(high_freq)
    features['liveness'] = float(min(1.0, high_freq_var / 0.1))
    
    # Time signature
    # This is a simplification - accurate time signature detection is complex
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    if len(beats) > 0:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        if len(beat_times) > 8:
            # Analyze patterns in beat intervals to estimate time signature
            beat_intervals = np.diff(beat_times)
            autocorr = librosa.autocorrelate(beat_intervals)
            peak_idx = np.argmax(autocorr[1:]) + 1
            if 3 <= peak_idx <= 5:
                features['time_signature'] = peak_idx
            else:
                features['time_signature'] = 4  # Default to 4/4
        else:
            features['time_signature'] = 4  # Default to 4/4
    else:
        features['time_signature'] = 4  # Default to 4/4
    
    return features

def process_multiple_songs(file_paths):
    """
    Process multiple songs and return a DataFrame of features
    
    Parameters:
    file_paths (list): List of paths to audio files
    
    Returns:
    pd.DataFrame: DataFrame containing features for all songs
    """
    all_features = []
    
    for path in file_paths:
        try:
            features = extract_features(path)
            features['filename'] = path
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    return pd.DataFrame(all_features)

#%%
import os
# Load in files
file_names = np.array(os.listdir('/volumes/Seagate/mtg-jamendo-dataset/00'))
files = np.char.add('/volumes/Seagate/mtg-jamendo-dataset/00/', file_names)

#%% 
# One File
#extract_features('/volumes/Seagate/mtg-jamendo-dataset/00/433600.mp3')

#%% Extract Features
files_sublist = files[0:100]

features_df = process_multiple_songs(files_sublist)

#%% Save df to csv
date = datetime.today().strftime('%Y-%m-%d')
features_df.to_csv('csv/features_df_'+date+'.csv', index=False)

#%%
# Example usage
if __name__ == "__main__":
    # Example file paths - replace with your actual audio files
    file_paths = ["path_to_song1.mp3", "path_to_song2.mp3"]
    
    # Extract features from all songs
    features_df = process_multiple_songs(file_paths)
    
    # Display the results
    print(features_df)
    
    # Save to CSV
    features_df.to_csv("song_features.csv", index=False)
    

# This code demonstrates how to calculate all the high-level features you mentioned in your document using low-level features from Librosa. Here's a breakdown of which low-level features are used for each high-level feature:

# Danceability:
# Beat tracking (beat frames, tempo)
# Onset strength
# Beat regularity (variance in beat intervals)

# Energy:
# Root Mean Square (RMS) energy

# Key:
# Chroma features (tonal content)

# Loudness:
# Decibel levels from the Short-Time Fourier Transform (STFT)
# Mode (major or minor):
# Chroma features compared with major/minor profiles

# Speechiness:
# Spectral flatness
# Energy in speech frequency bands

# Acousticness:
# Spectral contrast
# MFCC features

# Instrumentalness:
# MFCC variance in vocal ranges

# Liveness:
# High-frequency variance (audience noise detection)

# Valence:
# Combination of mode, tempo, and spectral centroid

# Tempo:
# Onset strength and beat tracking

# Duration:
# Direct calculation from audio length
# Time signature:
# Beat analysis and autocorrelation of beat intervals

# Librosa supports extracting all these low-level features, which makes it ideal for your project. The approximations I've provided are similar to those used in commercial systems, though they may need adjustment based on your specific data.
# Some caveats to consider:

# Some features (like valence) are quite subjective and may require further refinement
# You might want to normalize the features across your dataset
# For production, you'll likely want to optimize this code for performance

# %%
#%%
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load dataset
songs_df = pd.read_csv('csv/features_df_2025_03_17.csv')

songs_df_processed = songs_df.drop('filename',axis=1)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(songs_df_processed)

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
recommend_song('drake_fps.mp3', songs_df, nn_model)
# %%
