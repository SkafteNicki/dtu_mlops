import requests
import streamlit as st

if __name__ == "__main__":
    # Streamlit UI
    st.title("Image Classification App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Button to classify the uploaded image
        if st.button("Classify"):
            # Send the image to the backend for classification
            files = {"file": ("image.jpg", uploaded_file.read())}
            backend_url = "http://127.0.0.1:8000/classify/"  # Replace with the actual backend URL
            response = requests.post(backend_url, files=files, timeout=60)

            # Display the result
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction for {result['filename']}: {result['prediction']}")
            else:
                st.error(f"Failed to classify image. Error: {response.text}")
