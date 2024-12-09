import os
import numpy as np
import pandas as pd
import librosa
# import essentia
# import essentia.standard as es
import keras

class MusicAnalyzer:
    def __init__(self, audio_path):
        """
        Initialize the Music Analyzer with the given audio file
        
        Args:
            audio_path (str): Path to the audio file
        """
        self.audio_path = audio_path
        
        # Load audio file
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.mood_detector_model = keras.models.load_model(filepath="Emotion_Voice_Detection_Model.h5")
        print(f"Model loaded Successfully")
        
        # Essentia audio loader
        # loader = es.MonoLoader(filename=audio_path)
        # self.audio = loader()

#FOR MOOD DETECTION.
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label
        
    def extract_genre(self):
        """
        Extract genre using multiple classification methods
        
        Returns:
            dict: Genre classifications with confidence scores
        """
        # Placeholder for genre classification
        # Note: Actual implementation would require pre-trained models from MuMu or FMA
        genre_results = {
            'top_genres': [
                {'name': 'Rock', 'confidence': 0.85},
                {'name': 'Alternative', 'confidence': 0.12},
                {'name': 'Indie', 'confidence': 0.03}
            ]
        }
        return genre_results
    
    def detect_mood(self):
        """
        Detect mood using Essentia's mood classifier
        
        Returns:
            dict: Mood classifications
        """
        mfccs = np.mean(librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=40).T, axis=0)
        print(f"MFCC:{mfccs}")
        # Add batch and channel dimensions
        x = np.expand_dims(mfccs, axis=0)  # Add batch dimension (batch_size = 1)
        x = np.expand_dims(x, axis=-1)     # Add channel dimension (channels = 1)
        print(f"After MFCC: {x}")
        predictions = self.mood_detector_model(x)

        return self.convert_class_to_emotion(predictions)
    
    def identify_instruments(self):
        """
        Identify instruments in the track
        
        Returns:
            list: Detected instruments with confidence
        """
        # Placeholder for instrument detection
        # Actual implementation would use advanced ML models
        instruments = [
            {'name': 'Electric Guitar', 'confidence': 0.9},
            {'name': 'Drums', 'confidence': 0.85},
            {'name': 'Bass Guitar', 'confidence': 0.7}
        ]
        return instruments
    
    def extract_musical_features(self):
        """
        Extract key musical features
        
        Returns:
            dict: Musical features including key and BPM
        """
        # Estimate key
        # key = es.KeyDetection()(self.audio)[0]
        
        # Estimate tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        
        return {
            # 'key': key,
            'bpm': tempo
        }
    
    def perform_sentiment_analysis(self):
        """
        Perform sentiment analysis on audio
        
        Returns:
            dict: Sentiment analysis results
        """
        # Placeholder for sentiment analysis
        # Would typically use NLP models trained on music lyrics
        return {
            'overall_sentiment': 'Positive',
            'sentiment_score': 0.75
        }
    
    def transcribe_lyrics(self):
        """
        Transcribe lyrics with high accuracy
        
        Returns:
            dict: Transcription results
        """
        # Placeholder for lyrics transcription
        # Would use advanced speech recognition models
        return {
            'transcription': 'Partial lyrics transcription...',
            'confidence': 0.87
        }
    
    def analyze(self):
        """
        Perform comprehensive music analysis
        
        Returns:
            dict: Comprehensive analysis results
        """
        analysis_results = {
            # 'genre': self.extract_genre(),
            'mood': self.detect_mood(),
            # 'instruments': self.identify_instruments(),
            # 'musical_features': self.extract_musical_features(),
            # 'sentiment': self.perform_sentiment_analysis(),
            # 'lyrics': self.transcribe_lyrics()
        }
        return analysis_results
    
    def save_to_csv(self, results):
        """
        Save analysis results to CSV
        
        Args:
            results (dict): Analysis results
        """
        # Flatten and prepare data for CSV
        # df = pd.DataFrame([{
        #     'Genre': results['genre']['top_genres'][0]['name'],
        #     'Genre_Confidence': results['genre']['top_genres'][0]['confidence'],
        #     'Mood_Happy': results['mood']['happy'],
        #     'Mood_Sad': results['mood']['sad'],
        #     'Key': results['musical_features']['key'],
        #     'BPM': results['musical_features']['bpm'],
        #     'Sentiment': results['sentiment']['overall_sentiment'],
        #     'Sentiment_Score': results['sentiment']['sentiment_score']
        # }])

        df = pd.DataFrame([{
            'Audio': self.audio_path,
            'Mood': results['mood']
        }])
        
        # Save to CSV
        output_filename = os.path.splitext(self.audio_path)[0] + '_analysis.csv'
        df.to_csv(output_filename, index=False)
        print(f"Analysis saved to {output_filename}")

def main(audio_file_path):
    """
    Main function to run music analysis
    
    Args:
        audio_file_path (str): Path to the audio file
    """
    try:
        analyzer = MusicAnalyzer(audio_file_path)
        results = analyzer.analyze()
        print(results)
        
        # Print results
        print("Comprehensive Music Analysis Results:")
        for key, value in results.items():
            print(f"{key.capitalize()}: {value}")
        
        # Save to CSV
        analyzer.save_to_csv(results)
    
    except Exception as e:
        print(f"Error analyzing music file: {e}")

if __name__ == "__main__":
    # Replace with your audio file path
    main("A Beacon of Hope.mp3")