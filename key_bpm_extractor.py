import essentia
from essentia.standard import MonoLoader, KeyExtractor
import librosa
import numpy as np

class KeyBPMExtractor:
    def __init__(self, file_path):
        """
        Initialize the AudioAnalyzer with a specific audio file.
        
        Args:
            file_path (str): Path to the audio file
        """
        self.file_path = file_path
        self.audio = None
        self.sr = None
        self._load_audio()
    
    def _load_audio(self):
        """
        Load audio using Essentia's MonoLoader and librosa.
        """
        try:
            # Load with Essentia
            loader = MonoLoader(filename=self.file_path)
            self.audio = loader()
            
            # Load with librosa to get sample rate
            _, self.sr = librosa.load(self.file_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {self.file_path}: {e}")
            raise
    
    def extract_key(self):
        """
        Extract the musical key of the audio file.
        
        Returns:
            str: Formatted key and scale (e.g., "C Major")
        """
        try:
            key_extractor = KeyExtractor()
            key, scale, _ = key_extractor(self.audio)
            return f"{key} {scale}"
        except Exception as e:
            print(f"Error extracting key for {self.file_path}: {e}")
            return "Key Not Found"
    
    def extract_bpm(self):
        """
        Extract the tempo (BPM) of the audio file.
        
        Returns:
            int: Rounded beats per minute
        """
        try:
            tempo, _ = librosa.beat.beat_track(y=self.audio, sr=self.sr)
            return round(float(tempo))
        except Exception as e:
            print(f"Error extracting BPM for {self.file_path}: {e}")
            return 0
    
    def analyze(self):
        """
        Perform complete audio analysis.
        
        Returns:
            dict: Dictionary containing analysis results
        """
        return {
            "Filename": self.file_path.split('/')[-1],
            "Key": self.extract_key(),
            "BPM": self.extract_bpm()
        }

# Example usage
def main():
    # Example file path (replace with your actual path)
    file_path = 'A Beacon of Hope.mp3'
    
    # Create analyzer instance
    analyzer = KeyBPMExtractor(file_path)
    
    # Get analysis results
    results = analyzer.analyze()
    
    # Print results
    print("Audio Analysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    main()