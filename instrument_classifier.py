import os
import json
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

class AudioInstrumentClassifier:
    def __init__(self,instrument_model_path,model_json_path,embedding_model_path):

        self.models = []
        self.models_names = []
        self.metadatas = []
        self.instrument_model_path = instrument_model_path
        self.model_json_path = model_json_path
        self.embedding_model_path = embedding_model_path

    def load_instrument_model(self):

        # Load Instrument Prediction Model and its Metadata Path
        model_path = self.instrument_model_path
        metadata_path = self.model_json_path

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Prepare model
        model = TensorflowPredict2D(graphFilename=model_path)

        return model,metadata

    def load_embeddings(self,audio):

      embedding_model_path = self.embedding_model_path
      embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=embedding_model_path, output="PartitionedCall:1")
      embeddings = embedding_model(audio)

      return embeddings

    def predict(self, audio_path, sample_rate=16000):

        # Load audio
        loader = MonoLoader(sampleRate=sample_rate, filename=audio_path)
        audio = loader()

        #Load embeddings from Discog model.
        embeddings = self.load_embeddings(audio)

        #Load Instrument Model and MetaData.
        model,metadata = self.load_instrument_model()

        predictions = model(embeddings)

        #Take Mean across-each timeStamp
        predictions = predictions.mean(axis=0)
        results = {}

        for label, probability in zip(metadata['classes'], predictions):
          results[label] = float(f'{100 * probability:.1f}')

        return results

    def print_predictions(self, predictions):

      print(f"\nInstrument Model Predictions:")
      for label, probability in predictions.items():
          print(f'{label}: {probability}%')

# Example usage
def main():
    # Create classifier instance
    classifier = AudioInstrumentClassifier(instrument_model_path="/content/mtg_jamendo_instrument-discogs-effnet-1.pb",
                                           model_json_path="/content/mtg_jamendo_instrument-discogs-effnet-1.json",
                                           embedding_model_path="/content/discogs-effnet-bs64-1.pb")

    # Predict for an audio file
    audio_path = '/content/sad-dramatic-piano-sad-alone-drama-262415.mp3'
    predictions = classifier.predict(audio_path)
    print(predictions)

    # Print predictions
    classifier.print_predictions(predictions)

main()