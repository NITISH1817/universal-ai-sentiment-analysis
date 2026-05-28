import streamlit as st
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):

    text = text.lower()

    words = text.split()

    filtered_words = []

    for word in words:

        if word not in stopwords.words('english'):

            lemma_word = lemmatizer.lemmatize(word)

            filtered_words.append(lemma_word)

    return " ".join(filtered_words)

# Streamlit UI
st.set_page_config(
    page_title="Universal AI Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Universal AI Sentiment Analysis")

st.write("Enter multiple statements (one per line)")

# Multi-line input
user_input = st.text_area(
    "Enter statements",
    height=300
)

# Analyze button
if st.button("Analyze Statements"):

    if user_input.strip() == "":

        st.warning("Please enter some statements")

    else:

        # Split lines
        statements = user_input.splitlines()

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        results = []

        # Analyze each statement
        for statement in statements:

            if statement.strip() != "":

                cleaned_text = preprocess(statement)

                vectorized_text = vectorizer.transform([cleaned_text])

                prediction = model.predict(vectorized_text)

                probability = model.predict_proba(vectorized_text)

                confidence = max(probability[0]) * 100

                # Positive
                if prediction[0] == 1:

                    sentiment = "Positive 😊"

                    positive_count += 1

                # Negative
                elif prediction[0] == 0:

                    sentiment = "Negative 😞"

                    negative_count += 1

                # Neutral
                else:

                    sentiment = "Neutral 😐"

                    neutral_count += 1

                results.append({
                    "statement": statement,
                    "sentiment": sentiment,
                    "confidence": f"{confidence:.2f}%"
                })

        # Display results
        st.subheader("Results")

        for result in results:

            st.write("Statement:", result["statement"])

            st.write("Sentiment:", result["sentiment"])

            st.write("Confidence:", result["confidence"])

            st.markdown("---")

        # Final counts
        st.subheader("Final Summary")

        st.write(f"✅ Positive Statements: {positive_count}")

        st.write(f"❌ Negative Statements: {negative_count}")

        st.write(f"😐 Neutral Statements: {neutral_count}")

        # Find majority sentiment
        maximum = max(
            positive_count,
            negative_count,
            neutral_count
        )

        if maximum == positive_count:

            st.success("Major Sentiment: Positive 😊")

        elif maximum == negative_count:

            st.error("Major Sentiment: Negative 😞")

        else:

            st.warning("Major Sentiment: Neutral 😐")