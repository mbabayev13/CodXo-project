import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
# Load the YOLO model
model = YOLO(r"C:\Users\User\OneDrive\Desktop\New project\best.pt")

# Custom class names mapping (only change "cars" to "car")
class_names_mapping = {
    "cars": "car",  # Replace "cars" with "car"
}

# Streamlit app title
st.title("Car Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Detecting objects...")

    # Perform object detection
    results = model([image])

    # Initialize a list to store detection information
    detection_data = []

    # Process and display results without annotations on the image
    for result in results:
        # Iterate over detected objects
        for obj in result.boxes:
            # Extract class name, confidence score, and bounding box coordinates
            class_id = int(obj.cls[0])  # Class ID
            original_class_name = result.names[class_id]  # Original class name from model
            # Replace "cars" with "car" using the mapping
            class_name = class_names_mapping.get(original_class_name, original_class_name)  
            confidence = float(obj.conf[0].item()) * 100  # Convert Tensor to float and get percentage
            
            # Append detection info to the list
            detection_data.append({
                'Class': class_name,
                'Confidence (%)': round(confidence, 2),  # Round confidence
                'Bounding Box': obj.xyxy[0].tolist()  # Bounding box coordinates (optional)
            })
        
        # Draw bounding boxes on the image (without labels and confidence scores)
        result_image = np.array(image)  # Convert PIL image to NumPy array
        for obj in result.boxes:
            # Draw the bounding boxes on the image without labels
            bbox = obj.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)
            result_image = cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the result image with bounding boxes (no labels or confidence)
        st.image(result_image, caption='Detected Objects (No Labels)', use_column_width=True)

    # Display detection data in a table
    if detection_data:
        st.write("### Detected Objects and Confidence")
        detection_df = pd.DataFrame(detection_data)
        st.dataframe(detection_df)  # Display the table

    st.write("Detection complete.")
