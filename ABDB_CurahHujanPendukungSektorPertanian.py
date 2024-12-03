import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and dataset
MODEL_PATH = "decision_tree_model.pkl"
DATA_PATH = "weather_classification_data.csv"

# Function to load the model
def load_model():
    return joblib.load(MODEL_PATH)

# Function to load the dataset
def load_dataset():
    return pd.read_csv(DATA_PATH)

# Main application
def main():
    st.title("Weather Classification using Decision Tree")

    # Load the model and dataset
    model = load_model()
    df = load_dataset()

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Exploratory Data Analysis
    st.subheader("EDA")
    st.write("**Dataset Info:**")
    buffer = []
    df.info(buf=buffer)
    st.text("\n".join(buffer))

    st.write("**Summary Statistics:**")
    st.write(df.describe())

    st.write("**Class Distribution:**")
    class_dist = df['WeatherType'].value_counts()
    st.bar_chart(class_dist)

    # Preprocessing steps
    st.subheader("Preprocessing")
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna('unknown')

    label_encoder = LabelEncoder()
    df['WeatherType'] = label_encoder.fit_transform(df['WeatherType'])
    df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

    st.write("**Transformed Dataset:**")
    st.write(df_encoded.head())

    # Feature selection and prediction
    st.subheader("Make a Prediction")

    user_input = {}
    for feature in df_encoded.drop(columns=['WeatherType']).columns:
        user_input[feature] = st.number_input(f"Input {feature}", value=0.0)

    user_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(user_df)
        st.write("**Prediction Result:**")
        st.write(label_encoder.inverse_transform(prediction)[0])

    # Visualization of decision tree
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 10))
    from sklearn.tree import plot_tree
    plot_tree(model, feature_names=df_encoded.drop(columns=['WeatherType']).columns, class_names=label_encoder.classes_, filled=True, ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
