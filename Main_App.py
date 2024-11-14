import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pygam import LinearGAM


# Function to convert image to string with improved preprocessing
def image_to_text(image):
    # Convert to grayscale if the image is not already in grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)

    # Use Tesseract to extract text
    custom_config = r'--oem 3 --psm 6'  # OEM 3: Default, PSM 6: Assume a single uniform block of text
    text = pytesseract.image_to_string(thresh_image, config=custom_config)
    
    # Optionally, clean the extracted text (remove unwanted characters, etc.)
    text = ' '.join(text.split())  # Remove extra spaces
    return text



# Function to uncurve text
def uncurve_text(image, n_splines=5):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    black_pixels = np.column_stack(np.where(thresh == 0))
    leftmost_x, rightmost_x = np.min(black_pixels[:, 1]), np.max(black_pixels[:, 1])
    X = black_pixels[:, 1].reshape(-1, 1)
    y = thresh.shape[0] - black_pixels[:, 0]

    gam = LinearGAM(n_splines=n_splines)
    gam.fit(X, y)

    y_hat = gam.predict(np.linspace(leftmost_x, rightmost_x, num=rightmost_x - leftmost_x + 1))

    for i in range(leftmost_x, rightmost_x + 1):
        image[:, i, 0] = np.roll(image[:, i, 0], round(y_hat[i - leftmost_x] - thresh.shape[0] / 2))
        image[:, i, 1] = np.roll(image[:, i, 1], round(y_hat[i - leftmost_x] - thresh.shape[0] / 2))
        image[:, i, 2] = np.roll(image[:, i, 2], round(y_hat[i - leftmost_x] - thresh.shape[0] / 2))

    return image

# Main function for Streamlit app
def main():
    # Custom CSS for changing background color
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2ff;  /* light grayish blue */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("CurveFix: Text Alignment & Dewarping")

    # User selection: choose action before uploading file
    action = st.selectbox("Select Action", ["Align Text", "Extract Text"])

    # File uploader for images 
    uploaded_file = st.file_uploader("Upload a handwritten image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Check the uploaded file type
        if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
            try:
                # If the uploaded file is an image
                image = Image.open(uploaded_file)
                image_np = np.array(image)

                # Display the uploaded image
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Perform action based on user's choice
                if action == "Extract Text":
                    extracted_text = image_to_text(image_np)
                    st.subheader("Extracted Text:")
                    st.write(extracted_text if extracted_text.strip() else "No text extracted.")

                elif action == "Align Text":
                    processed_image = uncurve_text(image_np)
                    st.image(processed_image, caption='Aligned Image', use_column_width=True)

                    # Save the processed image in a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    cv2.imwrite(temp_file.name, processed_image)

                    # Provide download link for processed image
                    with open(temp_file.name, 'rb') as f:
                        st.download_button('Download Aligned Image', f, file_name="aligned_image.png")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

        elif uploaded_file.type == "application/pdf":
            # If the uploaded file is a PDF
            if action == "Extract Text":
                extracted_text = pdf_to_text(uploaded_file)
                st.subheader("Extracted Text from PDF:")
                st.write(extracted_text if extracted_text.strip() else "No text extracted.")
            else:
                st.error("Alignment is only supported for images, not PDFs.")
        else:
            st.error("Unsupported file format. Please upload a valid image or PDF.")

if __name__ == "__main__":
    main()
