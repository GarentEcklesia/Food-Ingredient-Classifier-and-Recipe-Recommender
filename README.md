# 🥗 Food Ingredient Classifier & Recipe Recommender

This is an end-to-end **AI-powered culinary assistant** that combines Computer Vision and Large Language Models (LLMs). It uses **MobileNetV2** to classify food ingredients from images and integrates with **Groq (Llama-3.3-70b)** to generate tailored cooking recipes based on the detected ingredients and **user-defined preferences**.

## 🚀 Live Demo

The application is deployed on Streamlit Cloud and is ready to use:

[**➡️ Click here to launch the Streamlit App**](https://food-ingredient-classifier-and-recipe-recommender-garent.streamlit.app/)

## 💡 Application Features

* **Instant Ingredient Recognition:** Classifies **51 different types** of fruits, vegetables, and common ingredients (e.g., Garlic, Ginger, Strawberry, Potato) with high accuracy.
* **Customizable AI Chef:** Users can **customize the prompt** to tailor the recipe generation. You can ask for specific requirements such as *"Make it vegan," "Low calorie,"* or *"Indonesian style."*
* **Computer Vision Backbone:** Uses **MobileNetV2**, optimized for speed and efficiency, making it ideal for web-based deployment.
* **Interactive UI:** Built with Streamlit for a seamless user experience, allowing image uploads, prompt editing, and displaying confidence scores.

## 🛠️ How It Works

1.  **Input:** The user uploads an image of a food item.
2.  **Vision Processing:** The image is processed by the **MobileNetV2** model.
3.  **Classification:** The model predicts the ingredient class (e.g., "Banana") with a specific confidence score.
4.  **Prompt Engineering:** The predicted class is combined with the **user's custom instructions** (e.g., dietary restrictions, cooking style).
5.  **Generative AI:** This dynamic prompt is sent to the **Groq API**, where the **Llama-3** model generates a personalized recipe.

## 📊 Model Performance

The Computer Vision model was trained on a dataset of 51 food classes.

* **Test Accuracy:** 93.15%
* **Top-3 Accuracy:** 98.63% (The correct ingredient is in the top 3 predictions 98% of the time)
* **Test Loss:** 0.2231

### Classification Report Summary

| Metric | Score |
| :--- | :--- |
| **Precision (Weighted Avg)** | 94.0% |
| **Recall (Weighted Avg)** | 93.2% |
| **F1-Score (Weighted Avg)** | 93.0% |

## 💻 Tech Stack

* **Deep Learning Framework:** TensorFlow / Keras
* **Base Model:** MobileNetV2 (Transfer Learning)
* **LLM Provider:** Groq API (Model: `llama-3.3-70b-versatile`)
* **Web Framework:** Streamlit
* **Data Processing:** NumPy, Pandas, Pillow (PIL)
* **Deployment Platform:** Streamlit Cloud

## 📂 Dataset

The model was trained on the **Food Ingredient Dataset**, which contains images of 51 different classes of food items.

* **Source:** [Kaggle - Food Ingredient Dataset (51 Classes)](https://www.kaggle.com/datasets/sunnyagarwal427444/food-ingredient-dataset-51)

## ⚙️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/GarentEcklesia/Food-Ingredient-Classifier](https://github.com/GarentEcklesia/Food-Ingredient-Classifier)
    cd Food-Ingredient-Classifier
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    * Create a `.env` file
    * Add your file:
        ```bash
        API_KEY = "your-api-key"
        ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 📬 Contact

Garent Ecklesia - [garentecklesia45678@gmail.com](mailto:garentecklesia45678@gmail.com)

## 📝 License

This project is open-source and free to use for educational and research purposes.
