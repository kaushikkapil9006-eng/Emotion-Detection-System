
import streamlit as st
import pickle

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="😊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

with open("emotion_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("bow_vectorizer.pkl", "rb") as file:
    bow_vectorizer = pickle.load(file)

# ---------------- TITLE ----------------

st.title("😊 Emotion Detection System")

st.subheader(
    "Understand Emotions Through Text Using NLP & Machine Learning"
)

st.write(
    "This application uses Natural Language Processing "
    "and Machine Learning to identify emotions from text."
)

st.divider()

# ---------------- EMOTION DICTIONARY ----------------

emotion_icons = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "fear": "😨",
    "surprise": "😲"
}

# ---------------- USER INPUT ----------------

st.header("Enter Your Text")

user_text = st.text_area(
    "Write a sentence:",
    placeholder="Example: I am feeling very happy today...",
    height=150
)

# ---------------- PREDICTION ----------------

if st.button("Predict Emotion", type="primary", use_container_width=True):

    if user_text.strip() == "":
        st.warning("Please enter some text first.")

    else:

        # Convert text into Bag of Words features
        text_vector = bow_vectorizer.transform([user_text])

        # Predict emotion
        prediction = model.predict(text_vector)[0]

        # Predict probabilities
        probabilities = model.predict_proba(text_vector)[0]

        # Get emotion icon
        icon = emotion_icons.get(prediction.lower(), "💭")

        # Display result
        st.success("Emotion Prediction Completed!")

        st.divider()

        st.header("Predicted Emotion")

        st.markdown(
            f"""
            <div style="
                background-color: #e8f5e9;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
            ">
                <h1 style="color: #1b5e20;">
                    {icon} {prediction.upper()}
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # ---------------- PROBABILITY ----------------

        st.header("Emotion Probability")

        emotion_probability = {
            emotion: float(probability)
            for emotion, probability in zip(
                model.classes_,
                probabilities
            )
        }

        st.bar_chart(emotion_probability)

        # Confidence
        confidence = max(probabilities) * 100

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

# ---------------- SIDEBAR ----------------

st.sidebar.title("About Project")

st.sidebar.info(
    """
    **Emotion Detection System**

    This project uses:

    - Python
    - Natural Language Processing
    - Bag of Words
    - Logistic Regression
    - Scikit-learn
    - Streamlit

    It predicts six emotions:

    - Joy
    - Sadness
    - Anger
    - Love
    - Fear
    - Surprise
    """
)

st.sidebar.markdown("---")

st.sidebar.write("Developed using Python and Machine Learning.")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Emotion Detection System | NLP + Machine Learning | Streamlit"
)
