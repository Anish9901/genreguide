from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import os
import librosa
import csv
import pickle

path = os.getcwd()

def predictor(csv_path):
    features = pd.read_csv(csv_path)
    with open(f'{path}/model.pkl','rb') as f:
        model = pickle.load(f)
    scaled_features = model['scaler'].transform(features)
    pred = model['svm'].predict(scaled_features)
    return pred[0]

class Features:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr
    
    def get_chroma_stft(self):
        # Chroma features represent the distribution of energy in different pitch classes.
        chroma_stft = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        chroma_stft_mean = np.mean(chroma_stft)
        chroma_stft_var = np.var(chroma_stft)
        return chroma_stft_mean, chroma_stft_var

    def get_rms(self):
        # RMS energy is the root mean square of the amplitude of the audio signal.
        # It is a measure of the average energy of the signal.
        rms = librosa.feature.rms(y=self.y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        return rms_mean, rms_var

    def get_spectral_centroid(self):
        # Spectral centroid is the weighted average of the frequencies present in the signal, 
        # with their magnitudes as the weights. It is a measure of the brightness of the signal. 
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)
        return spectral_centroid_mean, spectral_centroid_var

    def get_spectral_bandwidth(self):
        # Spectral bandwidth is a measure of the width of the frequency range over which the energy is distributed. 
        # It is a measure of the spread of the signal.
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)
        return spectral_bandwidth_mean, spectral_bandwidth_var

    def get_rolloff(self):
        # Spectral rolloff is the frequency below which a specified percentage of the total spectral energy lies. 
        # It is a measure of the shape of the signal. 
        rolloff = librosa.feature.spectral_rolloff(y=self.y+0.01, sr=self.sr)[0]
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)
        return rolloff_mean, rolloff_var

    def get_zero_crossing_rate(self):
        # Zero crossing rate is the rate at which the signal changes sign.
        # It is a measure of the number of times the signal crosses the zero-axis.
        zero_crossing_rate = librosa.feature.zero_crossing_rate(self.y)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        return zero_crossing_rate_mean, zero_crossing_rate_var

    def get_harmony(self):
        # Harmony is simultaneously occurring frequencies, pitches, or chords.
        harmony = librosa.effects.harmonic(self.y)
        harmony_mean = np.mean(harmony)
        harmony_var = np.var(harmony)
        return harmony_mean, harmony_var

    def get_perceptr(self):
        # The pecussive elements of the signal.
        perceptr = librosa.effects.percussive(self.y)
        perceptr_mean = np.mean(perceptr)
        perceptr_var = np.var(perceptr)
        return perceptr_mean, perceptr_var

    def get_tempo(self):
        # Tempo is the speed or pace of the music. It is measured in beats per minute (BPM).
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        tempo, _ = librosa.beat.beat_track(y=y_percussive)
        return tempo

    def get_mfcc(self):
        # These represent the shape of the spectral envelope of a signal.
        n_mfcc = 20 # Number of MFCC coefficients to calculate depending on the training dataset

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr)

        # Aggregate MFCC features into a single feature vector
        mfcc_features = _format_mfcc(n_mfcc, mfccs)
        return mfcc_features

def _format_mfcc(n_mfcc, mfccs):
    mfcc_features = []
    for i in range(n_mfcc):
        mfcc_features.append(np.mean(mfccs[i]))
        mfcc_features.append(np.var(mfccs[i]))
    mfcc_features = np.array(mfcc_features)
    return mfcc_features

def write_features_to_csv(features, file_name):
    csv_filepath = f'{path}/uploads/csv/{file_name}.csv'
    header = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
       'spectral_centroid_mean', 'spectral_centroid_var',
       'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
       'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
       'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
       'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
       'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
       'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
       'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
       'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
       'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
       'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
       'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']
    with open(csv_filepath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerow(features)
    return csv_filepath

def extract_features(audio_file_path, audio_file_name):
    y, sr = librosa.load(audio_file_path)
    # Preprocess audio file
    # Extract features
    ft = Features(y, sr)
    chroma_stft_mean, chroma_stft_var = ft.get_chroma_stft()
    rms_mean, rms_var = ft.get_rms()
    spectral_centroid_mean, spectral_centroid_var = ft.get_spectral_centroid()
    spectral_bandwidth_mean, spectral_bandwidth_var = ft.get_spectral_bandwidth()
    rolloff_mean, rolloff_var = ft.get_rolloff()
    zero_crossing_rate_mean, zero_crossing_rate_var = ft.get_zero_crossing_rate()
    harmony_mean, harmony_var = ft.get_harmony()
    perceptr_mean, perceptr_var = ft.get_perceptr()
    tempo = ft.get_tempo()
    mfcc = ft.get_mfcc()
    features = np.array([chroma_stft_mean, chroma_stft_var, rms_mean, rms_var,
       spectral_centroid_mean, spectral_centroid_var,
       spectral_bandwidth_mean, spectral_bandwidth_var, rolloff_mean,
       rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var,
       harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo, *mfcc])
    file_name_with_ext=audio_file_name.split('/')[-1]
    file_name = file_name_with_ext.split('.')[0]
    csv_file_path=write_features_to_csv(features, file_name)
    return csv_file_path