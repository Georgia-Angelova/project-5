
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import requests
import io

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("Upload a handwritten digit image and AI will try to recognize it.")

# Simple model loading with fallback
@st.cache_resource
def load_model():
    try:
        # Try to load pre-trained model
        # Using sklearn's built-in digits dataset
        from sklearn.datasets import load_digits
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split

        digits = load_digits()
        X = digits.images.reshape((len(digits.images), -1)) / 16.0
        y = digits.target

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )

        model.fit(X_train, y_train)

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load model
model = load_model()

# Image upload
st.subheader("Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image containing a handwritten digit",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Open uploaded image
        image = Image.open(uploaded_file).convert("L")

        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )

        # Preprocess image
        image = image.resize((8, 8))

        image_array = np.array(image)

        # Convert image to the format used by sklearn
        image_array = 255 - image_array
        image_array = image_array / 255.0

        image_array = image_array * 16.0

        image_array = image_array.reshape(1, -1)

        # Predict digit
        if model is not None:
            prediction = model.predict(image_array)

            st.success(
                f"Recognized Digit: {prediction[0]}"
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(image_array)[0]

                st.subheader("Prediction Confidence")

                st.bar_chart(
                    {
                        str(i): float(probabilities[i])
                        for i in range(10)
                    }
                )

    except Exception as e:
        st.error(f"Error processing image: {e}")
