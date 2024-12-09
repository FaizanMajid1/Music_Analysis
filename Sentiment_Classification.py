import streamlit as st
import librosa
from spleeter.separator import Separator
import whisper
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

token = "dummy"
login(token=token,add_to_git_credential=True)

# Title of the app
st.title("Audio Analysis and Transcription 🎵")

# Instructions for the user
st.write("Upload an MP3 file to analyze BPM, extract vocals, and transcribe lyrics.")

# File uploader
uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

# Load LLaMA model and tokenizer
@st.cache_resource
def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/LLaMA-2-7b-hf")
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_llama_model()

def analyze_sentiment_with_llama(lyrics):
    # Create the prompt for sentiment analysis
    prompt = f"Analyze the sentiment of the following text and describe it in one sentence:\n\n{lyrics}"

    # Tokenize and generate the output
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)
    
    # Decode and extract sentiment
    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentiment

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.mp3", "wb") as f:
        f.write(uploaded_file.read())

    # Load the audio file using librosa
    try:
        y, sr = librosa.load("temp.mp3", sr=None)

        # Detect the BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Convert tempo to a Python float and round it
        rounded_tempo = round(float(tempo))

        # Display BPM results
        st.success(f"The BPM of the uploaded audio file is: **{rounded_tempo} BPM**")

        # Separate vocals using Spleeter
        st.info("Separating vocals from the audio...")
        separator = Separator("spleeter:2stems")  # 2 stems: vocals and accompaniment
        separator.separate_to_file("temp.mp3", "output")

        # Load the separated vocals file
        vocals_path = os.path.join("output", "temp", "vocals.wav")
        if os.path.exists(vocals_path):
            st.success("Vocals extracted successfully!")

            # Transcribe vocals using Whisper
            st.info("Transcribing vocals to lyrics...")
            model = whisper.load_model("base")
            result = model.transcribe(vocals_path)
            lyrics = result["text"]

            # Display the transcribed lyrics
            st.success("Transcription completed!")
            st.write("**Extracted Lyrics:**")
            st.write(lyrics)

            # Pass lyrics to LLaMA model for sentiment analysis
            sentiment = analyze_sentiment_with_llama(lyrics)
            st.write("**Sentiment Analysis:**")
            st.write(sentiment)

        else:
            st.error("Failed to extract vocals. Please check the audio file.")

    except Exception as e:
        st.error(f"An error occurred while processing the audio: {e}")

else:
    st.info("Please upload an MP3 file to proceed.")

# Optional: Delete temporary files after processing
if os.path.exists("temp.mp3"):
    os.remove("temp.mp3")
if os.path.exists("output"):
    import shutil
    shutil.rmtree("output")
