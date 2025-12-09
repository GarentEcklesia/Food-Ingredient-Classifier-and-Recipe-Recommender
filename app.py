import streamlit as st
import numpy as np
import cv2
import json
import pickle
import tensorflow as tf
from PIL import Image
from groq import Groq
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

st.set_page_config(
    page_title="Food Ingredient Recipe Generator",
    page_icon="🍳",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Make text input more visible */
    input[type="text"] {
        border: 2px solid black !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 16px !important;
        background-color: #f9f9f9 !important;
        color: #000 !important;
    }

    /* Optional: adjust the text area too */
    textarea {
        border: 2px solid black !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 16px !important;
        background-color: #f9f9f9 !important;
        color: #000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# LOAD MODEL & METADATA
# ---------------------------
@st.cache_resource
def load_model_and_metadata():
    model = tf.keras.models.load_model('food_ingredient_classifier.h5')

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('model_config.json', 'r') as f:
        config = json.load(f)

    return model, label_encoder, config


# ---------------------------
# PREPROCESS IMAGE
# ---------------------------
def preprocess_image(image, img_size=224):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ---------------------------
# PREDICT TOP K
# ---------------------------
def predict_ingredients(image, model, label_encoder, img_size=224, top_k=5):
    img_processed = preprocess_image(image, img_size)
    predictions = model.predict(img_processed, verbose=0)

    top_k_idx = np.argsort(predictions[0])[-top_k:][::-1]

    results = []
    for idx in top_k_idx:
        results.append({
            "idx": int(idx),
            "ingredient": label_encoder.inverse_transform([idx])[0],
            "confidence": float(predictions[0][idx])
        })

    return results, predictions[0]


# ---------------------------
# RECIPE GENERATION
# ---------------------------
def generate_recipe(ingredients, confidences, user_instruction=""):
    api_key = os.getenv("API_KEY")
    if not api_key:
        return "❌ Missing GROQ_API_KEY in .env file."

    ingredient_list = "\n".join(
        [f"- {ing} ({conf*100:.1f}%)" for ing, conf in zip(ingredients, confidences)]
    )

    prompt = f"""
You are a helpful cooking assistant. 

Detected Ingredients:
{ingredient_list}

User Instruction: {user_instruction}

Please provide:
1. Recipe name
2. Additional ingredients needed
3. Step-by-step instructions
4. Cooking time
5. Serving suggestions
"""

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating recipe: {str(e)}"


# ===========================
#        MAIN UI
# ===========================
def main():

    st.title("🍳 Food Ingredient Recipe Generator")
    st.caption("Detect ingredients → Generate recipe → Enjoy cooking!")

    with st.sidebar:
        st.header("📦 Model Information")
        try:
            model, label_encoder, config = load_model_and_metadata()
            st.success("Model Loaded Successfully!")

            st.metric("Test Accuracy", f"{config['test_accuracy']*100:.1f}%")
            st.metric("Total Classes", config["num_classes"])

            if st.checkbox("Show Class Label Mapping"):
                st.json({i: lbl for i, lbl in enumerate(label_encoder.classes_)})

        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return

        top_k = st.slider("Top-K Predictions", 1, 10, 5)

    # Columns
    col1, col2 = st.columns([1, 1])

    # ---------------------------
    # LEFT SIDE: IMAGE INPUT
    # ---------------------------
    with col1:
        st.header("📸 Upload or Select Image")

        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        st.subheader("Or choose example:")
        example_paths = [
            "examples/Apple.jpg",
            "examples/Spinach.jpg",
            "examples/Garlic.jpg",
            "examples/Bitter Gourd.jpg"
        ]

        example_cols = st.columns(4)
        for i, col in enumerate(example_cols):
            with col:
                if os.path.exists(example_paths[i]):
                    img = Image.open(example_paths[i])
                    st.image(img)
                    if st.button(f"Use Example {i+1}"):
                        st.session_state["image"] = img

        if uploaded_file:
            st.session_state["image"] = Image.open(uploaded_file)
            st.image(st.session_state["image"], caption="Uploaded Image")

        if "image" in st.session_state:
            if st.button("🔍 Detect Ingredients"):
                with st.spinner("Analyzing image..."):
                    preds, raw_probs = predict_ingredients(
                        st.session_state["image"], model,
                        label_encoder, config["img_size"], top_k
                    )
                    st.session_state["predictions"] = preds
                    st.session_state["raw_probs"] = raw_probs

    # ---------------------------
    # RIGHT SIDE: RESULTS
    # ---------------------------
    with col2:
        st.header("🎯 Prediction Results")

        if "predictions" in st.session_state:
            preds = st.session_state["predictions"]

            # Show top-K results
            for i, p in enumerate(preds, 1):
                st.metric(f"{i}. {p['ingredient']}", f"{p['confidence']*100:.1f}%")
                st.progress(p["confidence"])

            # Probability distribution visualization
            st.subheader("📊 Full Probability Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(len(st.session_state["raw_probs"])), st.session_state["raw_probs"])
            ax.set_title("Class Probability Distribution")
            ax.set_xlabel("Class Index")
            ax.set_ylabel("Probability")
            st.pyplot(fig)

            st.markdown("---")

            # Add user instruction input
            user_instruction = st.text_input(
                "💡 Add your instructions (e.g., vegan, less spicy, low-carb):", ""
            )

            if st.button("🍽️ Generate Recipe"):
                with st.spinner("Generating recipe..."):
                    ingredients = [p["ingredient"] for p in preds]
                    confs = [p["confidence"] for p in preds]
                    recipe = generate_recipe(ingredients, confs, user_instruction)
                    st.session_state["recipe"] = recipe

    # ---------------------------
    # RECIPE OUTPUT
    # ---------------------------
    if "recipe" in st.session_state:
        st.header("📖 Generated Recipe")
        st.markdown(st.session_state["recipe"])
        st.download_button("📥 Download Recipe", st.session_state["recipe"], "recipe.txt")


if __name__ == "__main__":
    main()

