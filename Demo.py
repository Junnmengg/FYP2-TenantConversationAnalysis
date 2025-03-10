# MWE Prediction
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from torch import nn
import string
from huggingface_hub import hf_hub_download
import os

# Emotion Prediction
import numpy as np
from datetime import datetime
from transformers import pipeline, AutoModelForSequenceClassification
from textblob import TextBlob 
import altair as alt

# Personality Trait Prediction
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import emoji

Username1 = os.getenv('USERNAME1')
Username2 = os.getenv('USERNAME2')
Token = os.getenv('TOKEN')

st.set_page_config(
    page_title="ChatPulse App",
    page_icon="🚀"
)

# Function to check if a token is punctuation
def is_punctuation(token):
    return all(char in string.punctuation for char in token)

# Define the RoBERTa-CRF Model
class RoBertaCRFModel(nn.Module):
    def __init__(self, model_name, num_labels_mwe):
        super(RoBertaCRFModel, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier_mwe = nn.Linear(self.roberta.config.hidden_size, num_labels_mwe)
        self.crf = CRF(num_labels_mwe, batch_first=True)

    def forward(self, input_ids, attention_mask, labels_mwe=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits_mwe = self.classifier_mwe(sequence_output)

        mask = attention_mask.bool()
        if labels_mwe is not None:
            labels_mwe = labels_mwe.clone()
            labels_mwe[labels_mwe == -100] = 0
            loss = -self.crf(logits_mwe, labels_mwe, mask=mask)
            return loss
        else:
            predictions = self.crf.decode(logits_mwe, mask=mask)
            return predictions

# Load tokenizer and model from Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model_repo = Username1  # Replace with your Hugging Face repo ID
    token = Token  # Replace with your token

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_auth_token=token)

    # Download model weights using Hugging Face Hub
    model_weights_path = hf_hub_download(repo_id=model_repo, filename="roberta_crf_model_weights.pth", use_auth_token=token)

    # Initialize the model
    model = RoBertaCRFModel('roberta-base', num_labels_mwe=len(mwe_label_to_id))

    # Load weights into the model
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()

    return model, tokenizer

# Map tag indices to labels
mwe_label_to_id = {'B-MWE': 0, 'I-MWE': 1, 'O': 2}
idx2tag = {v: k for k, v in mwe_label_to_id.items()}

# Load Emotion Model
@st.cache_resource
def load_emotion_model():
    model_name = Username2
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe_lr = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return pipe_lr

pipe_lr = load_emotion_model()

# Function for MWE Detection
def perform_mwe_detection(sentence, model, tokenizer):
    """
    Perform MWE detection using the RoBERTa-CRF model.

    Args:
        sentence (str): Input sentence to analyze.
        model (nn.Module): The loaded RoBERTa-CRF model.
        tokenizer (AutoTokenizer): Tokenizer for the RoBERTa-CRF model.

    Returns:
        table_data (list): Token-level predictions with tags.
        detected_mwes (list): Detected MWEs (multi-word expressions).
    """
    # Tokenize input sentence
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    )

    # Get input tensors
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Make predictions
    with torch.no_grad():
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)
        crf_predictions = predictions[0]

    # Decode tokens and predictions
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    current_mwe_tokens = []
    detected_mwes = []

    for token, pred_idx in zip(tokens, crf_predictions):
        if token in ['<s>', '</s>', '<pad>']:
            continue

        clean_token = token.lstrip("Ġ")
        pred_tag = idx2tag.get(pred_idx, 'O')

        # If punctuation, force label to "O"
        if is_punctuation(clean_token):
            pred_tag = 'O'

        # Collect MWEs
        if pred_tag in ['B-MWE', 'I-MWE']:
            current_mwe_tokens.append(clean_token)
        elif current_mwe_tokens:
            detected_mwes.append(' '.join(current_mwe_tokens))
            current_mwe_tokens = []

    # Capture remaining MWE tokens
    if current_mwe_tokens:
        detected_mwes.append(' '.join(current_mwe_tokens))

    return detected_mwes

# Function for Emotion Detection
# Function: Predict Emotions
def predict_emotions(docx, threshold=0.2):
    results = pipe_lr(docx)[0]
    filtered_results = [r['label'] for r in results if r['score'] >= threshold]
    return filtered_results

emotions_dict = {
    "LABEL_2": "Anger 😠", "LABEL_5": "Surprise 😮", "LABEL_3": "Disgust 🤮",
    "LABEL_0": "Enjoyment 😄", "LABEL_4": "Fear 😨", "LABEL_1": "Sadness 😔", "LABEL_6": "Neutral 😐"
}

# Function: Sentiment Polarity Score using TextBlob
def calculate_sentiment_polarity(sentence):
    sentiment_score = TextBlob(sentence).sentiment.polarity
    return round(sentiment_score, 2)

# Function: Sentiment Category based on Sentiment Polarity Score
def classify_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Load model and tokenizer
with st.spinner("Loading the model..."):
    model, tokenizer = load_model_and_tokenizer()

    st.title("Tenant Conversation Analysis 📊")
    # st.write(
    #     """
    #     This app detects **Multi-Word Expressions (MWEs)** in a sentence using a fine-tuned 
    #     **RoBERTa-CRF model**.
    #     """
    # )

    # Input box for user sentence
    with st.form(key='text_input_form'):
        raw_text = st.text_area("💬 Conversation", placeholder="Enter your text here...")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        # Perform MWE detection
        if raw_text.strip() == "":
            st.warning("⚠️ Please enter text before submitting!")
        else:
            conversations = [conv.strip() for conv in raw_text.strip().split("\n\n") if conv.strip()]
            final_conversations = []
            
            for conversation in conversations:
                if "/END" in conversation:
                    final_conversations.append(conversation.replace("/END", "").strip())  # Keep only the last valid sentence
                    break  # Stop processing further sentences
                else:
                    final_conversations.append(conversation)

            # Process only the valid conversations
            emotion_data = []
            for idx, conversation in enumerate(final_conversations, start=1):
                detected_mwes = perform_mwe_detection(conversation.lower(), model, tokenizer)
                detected_emotions = predict_emotions(conversation.lower())
                sentiment_polarity = calculate_sentiment_polarity(conversation)
                sentiment = classify_sentiment(sentiment_polarity)

                emotion_data.extend([(f"Sentence {idx}", emo) for emo in detected_emotions])
                
                highlighted_text = conversation
                for mwe in detected_mwes:
                    highlighted_text = highlighted_text.replace(mwe, f'<span style="border: 2px solid skyblue; padding: 2px; border-radius: 5px;">{mwe}</span>')
                
                st.markdown(f"<h5 style='color:skyblue;'>Sentence {idx}: </h5>", unsafe_allow_html=True)
                st.markdown(f"<p>💬 {highlighted_text}</p>", unsafe_allow_html=True)

                # Define score color for the border
                score_color = "green" if sentiment_polarity > 0 else "red" if sentiment_polarity < 0 else "gray"
                sentiment_colors = {
                    "Positive": "green",
                    "Negative": "red",
                    "Neutral": "gray"
                }

                # Generate emotions display without underlining
                detected_emotions_display = " | ".join([emotions_dict.get(label, "Unknown") for label in detected_emotions]) if detected_emotions else "No strong emotion detected."

                # Display everything inside a bordered box
                st.markdown(
                    f"""
                    <div style='border: 2px solid {score_color}; padding: 10px; border-radius: 8px; margin-top: 10px;'>
                        <p><strong>Detected Emotions:</strong> {detected_emotions_display}</p>
                        <p><strong>Sentiment Polarity Score:</strong> <span style='color:{score_color};'>{sentiment_polarity} ({sentiment})</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Add some spacing for aesthetics
                st.markdown("---")

            st.markdown(f"<h5 style='color:skyblue;'>Conversation Summary: </h5>", unsafe_allow_html=True)

            if emotion_data:
                df = pd.DataFrame(emotion_data, columns=["Sentence", "Emotion"])

                # Define emotion labels based on emotions_dict
                emotions_dict = {
                    "LABEL_2": "Anger 😠", "LABEL_5": "Surprise 😮", "LABEL_3": "Disgust 🤮",
                    "LABEL_0": "Enjoyment 😄", "LABEL_4": "Fear 😨", "LABEL_1": "Sadness 😔", "LABEL_6": "Neutral 😐"
                }

                # Convert detected labels to their corresponding emoji-based names
                df["Emotion"] = df["Emotion"].map(emotions_dict)

                # Ensure all emotions appear on the y-axis
                all_emotions = list(emotions_dict.values())  # Fixed y-axis order with emoji labels

                # Convert to categorical type to force Altair to show all emotions
                df["Emotion"] = pd.Categorical(df["Emotion"], categories=all_emotions, ordered=True)

                # Convert "Sentence 1" -> 1, "Sentence 2" -> 2 for simpler x-axis labels
                df["Sentence"] = df["Sentence"].str.extract(r'(\d+)').astype(int)

                unique_sentences = df["Sentence"].unique()

                # Define a base chart
                base = alt.Chart(df).encode(
                    x=alt.X("Sentence:O", sort=unique_sentences, title="Sentence"),
                    y=alt.Y("Emotion:N", sort=all_emotions, title="Emotion"),
                    tooltip=["Sentence", "Emotion"]
                )

                # Add emotion markers with 'X' symbols
                emotion_chart = base.mark_point(size=80, shape="diamond", color="gold")

                # Add horizontal grid lines for each emotion label
                horizontal_grid = alt.Chart(pd.DataFrame({"Emotion": all_emotions})).mark_rule(
                    strokeDash=[4, 4], color="lightgray"
                ).encode(
                    y=alt.Y("Emotion:N", sort=all_emotions)  # Grid lines at each emotion label
                )

                # Combine emotion markers
                emotion_chart = (horizontal_grid + emotion_chart).interactive()

            # Create a DataFrame for sentiment polarity visualization
            sentiment_df = pd.DataFrame({
                "Sentence": unique_sentences,
                "Sentiment Polarity Score": [calculate_sentiment_polarity(final_conversations[i-1]) for i in unique_sentences]
            })

            # Create an interactive line chart
            sentiment_chart = (
                alt.Chart(sentiment_df)
                .mark_line(color="gold")  # Line color
                .encode(
                    x=alt.X("Sentence:O", sort=unique_sentences, title="Sentence"),
                    y=alt.Y("Sentiment Polarity Score:Q", scale=alt.Scale(domain=[-1, 1]), title="Sentiment Polarity Score"),
                    tooltip=["Sentence", "Sentiment Polarity Score"]
                )
                + 
                alt.Chart(sentiment_df)
                .mark_point(shape="diamond", color="gold", size=80)  # Diamond shape and gold color
                .encode(
                    x=alt.X("Sentence:O", sort=unique_sentences, title="Sentence"),
                    y=alt.Y("Sentiment Polarity Score:Q", scale=alt.Scale(domain=[-1, 1]), title="Sentiment Polarity Score"),
                    tooltip=["Sentence", "Sentiment Polarity Score"]
                )
            ).interactive()       

            # Create two tabs for analysis
            tab1, tab2 = st.tabs(["Emotion Analysis", "Sentiment Polarity Analysis"])

            with tab1:
                st.altair_chart(emotion_chart, use_container_width=True)

            with tab2:
                st.altair_chart(sentiment_chart, use_container_width=True)

            st.markdown(f"<h5 style='color:skyblue;'>Conclusion: </h5>", unsafe_allow_html=True)

            # Get the last sentence analysis
            if final_conversations:
                last_sentence = final_conversations[-1]
                last_detected_emotions = predict_emotions(last_sentence.lower())
                last_sentiment_polarity = calculate_sentiment_polarity(last_sentence)

                # Check if "Enjoyment 😄" is in the detected emotions and polarity > 0
                problem_resolved = "✅" if "LABEL_0" in last_detected_emotions and last_sentiment_polarity > 0 else "❌"

                # Display problem resolution status without border
                st.markdown(f"###### 📞 Problem Resolution: {problem_resolved}", unsafe_allow_html=True)  
