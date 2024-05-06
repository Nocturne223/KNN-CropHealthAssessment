import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import smtplib
 
from PIL import Image
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Load Models
model1 = load_model('official-models/LettuceModel-2.h5')  # saved model from training
model2 = load_model('official-models/CauliflowerModel-2.h5')  # saved model from training
model3 = load_model('official-models/SugarcaneModel-2.h5')  # saved model from training
model4 = load_model('official-models/PepperModel-2.h5')  # saved model from training

# Define plant class names
Lettuce_names = ["lettuce_BacterialLeafSpot", "lettuce_BotrytisCrownRot", "lettuce_DownyMildew", "lettuce_Healthy"]
Cauliflower_names = ["cauliflower_BlackRot", "cauliflower_DownyMildew", "cauliflower_Healthy", "cauliflower_SoftRot"]
Sugarcane_names = ["sugarcane_Healthy", "sugarcane_Mosaic", "sugarcane_RedRot", "sugarcane_Rust"]
Pepper_names = ["pepper_Healthy", "pepper_CercosporaLeafSpot", "pepper_Fusarium", "pepper_Leaf_Curl"]

folder_path = "saved_images"

# Email Variable Definitions
sender_email = "fitnessapp96@gmail.com"
sender_password = "xxesgxqubzucxyco"  # Replace with the actual password
recipient_email = "e.albert223@gmail.com"
user_email = ""
feedback_text = "Feedback"
rating = 5
feedback_category = "User Interface"

# Define Variables Outside conditions
predicted_class = ""
recommendations_string = ""

# Define feedback email function
def send_feedback_email(sender_email, sender_password, recipient_email, feedback_text, rating, feedback_category):
    # Set up the SMTP server
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Create a secure connection to the SMTP server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()

    # Log in to the email account
    server.login(sender_email, sender_password)

    # Construct the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "Feedback Submission"

    # Compose the email body
    body = f"""
    Feedback Text: {feedback_text}
    Rating: {rating}
    Feedback Category: {feedback_category}
    """
    msg.attach(MIMEText(body, 'plain'))

    # Send the email
    server.send_message(msg)

    # Close the connection to the SMTP server
    server.quit()

# Define result email function
def send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendations):
    # Email configurations
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    
    # Create a secure connection to the SMTP server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
        
    # Log in to the email account
    server.login(sender_email, sender_password)
    
    # Construct the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = user_email
    msg['Subject'] = "Crop Health Assessment Result"
    
    body = f"""
    
    File Name:  {image_path.name}\n\n
    Predicted Disease Class: {predicted_class}\n\n
    Recommendations:\n{recommendations}
    
    """
        
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Send the email
    server.send_message(msg)
    st.success("Result sent successfully!")
    server.quit()
    
def get_recommendations_as_string(predicted_class):
    recommendations_string = ""
    recommendations_list = recommendations.get(predicted_class, [])
    
    if recommendations_list:
        recommendations_string += "Recommendations for " + predicted_class + ":\n"
        for recommendation in recommendations_list:
            recommendations_string += "- " + recommendation + "\n"
    else:
        recommendations_string += "No recommendations found for " + predicted_class + "."
    
    return recommendations_string

# Define recommendations for each class
recommendations = {
    "lettuce_BacterialLeafSpot": [
        "Use disease-resistant lettuce varieties whenever possible.",
        "Practice crop rotation with non-host crops to reduce the buildup of bacterial pathogens in the soil.",
        "Avoid overhead irrigation to minimize leaf wetness, as bacterial leaf spot thrives in moist conditions.",
        "Apply copper-based fungicides or bactericides according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "lettuce_BotrytisCrownRot": [
        "Practice proper spacing between plants to improve air circulation and reduce humidity levels around the lettuce crowns.",
        "Avoid overhead irrigation and water lettuce at the base to prevent water accumulation in the crown.",
        "Remove and destroy infected plants and debris to prevent the spread of the fungus.",
        "Apply fungicides containing active ingredients such as iprodione or boscalid to protect healthy plants from infection."
    ],
    "lettuce_DownyMildew": [
        "Plant lettuce varieties with genetic resistance to downy mildew if available.",
        "Ensure proper drainage to prevent waterlogging, as downy mildew thrives in wet conditions.",
        "Apply fungicides containing active ingredients such as metalaxyl, mandipropamid, or chlorothalonil according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "lettuce_Healthy": [
        "Maintain consistent soil moisture levels to ensure even growth and prevent stress.",
        "Provide adequate spacing between plants to promote air circulation and reduce the risk of foliar diseases.",
        "Monitor for signs of pests and diseases regularly, and take prompt action if detected.",
        "Implement good sanitation practices, including removing debris and weeds, to reduce the risk of disease spread."
    ],
    "cauliflower_BlackRot": [
        "Practice crop rotation with non-cruciferous crops to reduce the buildup of black rot pathogens in the soil.",
        "Remove and destroy infected plant debris to prevent the spread of the disease.",
        "Apply fungicides containing active ingredients such as copper or chlorothalonil according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "cauliflower_DownyMildew": [
        "Plant cauliflower varieties with genetic resistance to downy mildew if available.",
        "Ensure proper spacing between plants to improve air circulation and reduce humidity levels.",
        "Apply fungicides containing active ingredients such as mandipropamid, fosetyl-aluminum, or copper hydroxide according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "cauliflower_SoftRot": [
        "Practice proper sanitation and hygiene to prevent contamination of cauliflower heads during harvesting and handling.",
        "Harvest cauliflower heads at the correct maturity stage and handle them carefully to avoid bruising or injury.",
        "Store harvested cauliflower heads in cool, dry conditions to minimize the risk of soft rot development.",
        "Apply fungicides containing active ingredients such as iprodione or thiophanate-methyl to protect harvested cauliflower heads from soft rot during storage."
    ],
    "cauliflower_Healthy": [
        "Provide consistent moisture and fertility to support healthy growth and development.",
        "Monitor for signs of pests and diseases regularly, and take prompt action if detected.",
        "Implement good cultural practices, including proper spacing and soil management, to promote plant health and vigor."
    ],
    "sugarcane_Healthy": [
        "Implement proper irrigation and drainage practices to ensure adequate moisture without waterlogging.",
        "Monitor for signs of pests and diseases regularly, and take prompt action if detected.",
        "Implement good cultural practices, including weed control and soil management, to promote plant health and vigor."
    ],
    "sugarcane_Mosaic": [
        "Plant mosaic-resistant sugarcane varieties whenever possible.",
        "Use certified disease-free seed cane to minimize the risk of mosaic virus transmission.",
        "Implement strict sanitation practices to prevent the spread of mosaic virus within the sugarcane plantation.",
        "Control aphid populations, which can transmit mosaic virus, through insecticide applications or cultural practices."
    ],
    "sugarcane_RedRot": [
        "Use disease-free seed cane from reputable sources to prevent the introduction of red rot pathogens.",
        "Practice proper sanitation and hygiene to prevent the spread of red rot within the sugarcane plantation.",
        "Apply fungicides containing active ingredients such as mancozeb or thiophanate-methyl to protect healthy plants from infection."
    ],
    "sugarcane_Rust": [
        "Plant rust-resistant sugarcane varieties whenever possible.",
        "Ensure proper spacing between sugarcane rows to promote air circulation and reduce humidity levels, which can favor rust development.",
        "Apply fungicides containing active ingredients such as propiconazole or tebuconazole according to label instructions, especially during periods of high humidity or when rust symptoms first appear."
    ],
    "pepper_Healthy": [
        "Provide consistent moisture and fertility to support healthy growth and development.",
        "Monitor for signs of pests and diseases regularly, and take prompt action if detected.",
        "Implement good cultural practices, including proper spacing and soil management, to promote plant health and vigor.",
        "Use mulch to conserve soil moisture, suppress weeds, and maintain even soil temperatures around pepper plants."
    ],
    "pepper_CercosporaLeafSpot": [
        "Use disease-resistant pepper varieties whenever possible.",
        "Practice crop rotation with non-host crops to reduce pathogen buildup in the soil.",
        "Apply organic mulch around plants to prevent soil splash, which can spread the fungus.",
        "Avoid overhead irrigation to minimize leaf wetness and create drier conditions unfavorable for fungal growth.",
        "Apply copper-based fungicides according to label instructions, especially during periods of high humidity or when symptoms first appear."
    ],
    "pepper_Fusarium": [
        "Plant Fusarium-resistant pepper varieties if available.",
        "Practice crop rotation with non-host crops to reduce soilborne pathogen populations.",
        "Maintain optimal soil drainage to minimize waterlogging, as Fusarium pathogens thrive in wet conditions.",
        "Use soil solarization to reduce soilborne pathogens before planting.",
        "Apply biofungicides containing beneficial microorganisms to the soil to suppress Fusarium growth."
    ],
    "pepper_Leaf_Curl": [
        "Remove and destroy infected plants to prevent the spread of the disease to healthy plants.",
        "Avoid planting peppers in areas prone to high humidity or where water stagnates, as this can exacerbate leaf curl symptoms.",
        "Maintain proper spacing between plants to improve air circulation and reduce humidity levels around plants.",
        "Apply neem oil or horticultural oils to the foliage according to label instructions, as they may help suppress viral vectors like aphids."
    ]
    # Add recommendations for other classes similarly
}

# Function to display recommendations and predicted class
def display_recommendations(predicted_class):
    st.subheader("Recommendations:")
    for recommendation in recommendations.get(predicted_class, []):
        st.write(recommendation)

# Function to preprocess input image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize image to match model's input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to generate plot for SVM predictions
def generate_svm_plot(prediction_probabilities, class_names):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 6))
    plt.plot(prediction_probabilities[0], label=class_names, linestyle='-', marker='o', markersize=5)
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.grid(True, which='both', linestyle='--', linewidth=0.1)
    plt.legend(fontsize=12)
    
    # Display the plot
    st.pyplot()
    
    # Add explanation
    explanation = """
    This plot shows the probability distribution across different classes.
    Each class represents a potential disease or health status of the crop.
    The x-axis represents the classes, while the y-axis represents the probability of each class.
    The class with the highest probability is the predicted disease or health status.
    """
    st.write(explanation)

# Function to predict disease using Model
def predict_disease(model, image_path, names):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    disease_index = np.argmax(prediction)  # Get the index of the predicted class
    disease_class = names[disease_index]  # Fetch the class name using the index
    return prediction, disease_class  # Return prediction along with disease class

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Crop Health Assessment", "Feedback", "About Us"])

# Sidebar content
with st.sidebar:
    st.sidebar.subheader("Help & FAQ", divider='gray')

    # Help section
    st.sidebar.subheader("Help")
    st.sidebar.write("Welcome to the Help & FAQ section. If you have any questions or need assistance, please refer to the FAQs below or contact us for support.")

    # FAQ section
    st.sidebar.subheader("FAQs")

    # FAQ 1
    st.sidebar.write("Q: How do I use the app?")
    st.sidebar.write("A: To use the app, navigate to the 'Crop Health Assessment' tab and follow the instructions provided.")

    # FAQ 2
    st.sidebar.write("Q: What crops does the app support?")
    st.sidebar.write("A: The app currently supports lettuce, cauliflower, sugarcane, and pepper.")

    # FAQ 3
    st.sidebar.write("Q: How accurate are the predictions?")
    st.sidebar.write("A: The accuracy of predictions depends on various factors, including the quality of input data and the performance of the underlying machine learning models.")

    st.subheader('About Us', divider='gray')
    st.info(
        """
        Authors:
        
        Christian Jerome S. Detuya
        
        Albert James E. Mangcao
        
        Axel Bert E. Ramos
        """
    )

with tab1:
    st.title("Welcome to Crop Health Assessment App",False)
    st.subheader("", divider='gray')
    
    col1, col2 = st.columns(2)
    col1.image("screenshots/PPrediction1.jpeg")
    col2.image("screenshots/PPrediction2.jpeg")
    
    st.write("""
        Welcome to the Crop Health Assessment App! This application aims to assist you in analyzing the health of your crops using advanced machine learning techniques.

        ## How to Use Camera Method
        - Proceed to Crop Health Assessment Tab.
        - Select the "Camera" option to capture an image of your crop.
        - Click "Allow" in the Popup message to use your device camera.
        - Click "Take Photo" to capture an image of your crop.
        - Choose the type of crop from the dropdown menu.
        - Click on the "Submit" button to analyze the uploaded image.
        
        ## How to Use Upload Method
        - Proceed to Crop Health Assessment Tab.
        - Select the "Upload" option to upload an image of your crop.
        - Choose the image file from your device.
        - Choose the type of crop from the dropdown menu.
        - Click on the "Submit" button to analyze the uploaded image.

        ## Method
        You can either upload an image from your device or use your device's camera to capture an image directly.
 """)

with tab2:
    st.title("Crop Health Assessment")
    st.subheader("", divider='gray')

    # selecting method for health assessments
    st.subheader("SELECT A METHOD")
    pick = st.selectbox("Select Method", ('Upload', 'Camera'), label_visibility="hidden")

    if pick == 'Camera':
        st.subheader("Camera Input")
        plantpic = st.camera_input("Take a plant picture", label_visibility="hidden")
        
        st.subheader("Select A Plant")
        select = st.selectbox("Select Plant", ('Lettuce', 'Cauliflower', 'Sugarcane', 'Pepper'), label_visibility="hidden")
        user_email = st.text_input("(Optional)Enter recipient email address", help="Enter the email address where you want to receive the result.") 
        submit = st.button("Submit", use_container_width=True)
        
        if submit:
            if not plantpic:
                st.write("Please take a photo!")
            else:
                # Perform prediction and display results
                st.subheader("Prediction: ")
                
                if select == 'Lettuce':
                    # Predict Lettuce disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model1, image_path, Lettuce_names)
                    pred1 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred1, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Lettuce " + predicted_class + ":" 

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    st.subheader("Prediction Probabilities:")
                    generate_svm_plot(prediction, predicted_class)
    
                    recommendation = get_recommendations_as_string(predicted_class)

                    if not user_email == "":
                    # Call the function to send the result to the provided email
                        send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendation)
                        st.success("Results sent via Email.")
                        
                elif select == 'Cauliflower':
                    # Predict Cauliflower disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model2, image_path, Cauliflower_names)
                    pred2 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred2, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Cauliflower " + predicted_class + ":" 

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    st.subheader("Prediction Probabilities:")
                    generate_svm_plot(prediction, predicted_class)
    
                    recommendation = get_recommendations_as_string(predicted_class)

                    if not user_email == "":
                    # Call the function to send the result to the provided email
                        send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendation)
                        st.success("Results sent via Email.")
                        
                elif select == 'Sugarcane':
                    # Predict Sugarcane disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model3, image_path, Sugarcane_names)
                    pred3 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred3, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Sugarcane " + predicted_class + ":" 

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    st.subheader("Prediction Probabilities:")
                    generate_svm_plot(prediction, predicted_class)
    
                    recommendation = get_recommendations_as_string(predicted_class)

                    if not user_email == "":
                    # Call the function to send the result to the provided email
                        send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendation)
                        st.success("Results sent via Email.")

                elif select == 'Pepper':
                    # Predict Pepper disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model4, image_path, Pepper_names)
                    pred4 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred4, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Pepper " + predicted_class + ":" 

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    st.subheader("Prediction Probabilities:")
                    generate_svm_plot(prediction, predicted_class)
    
                    recommendation = get_recommendations_as_string(predicted_class)

                    if not user_email == "":
                    # Call the function to send the result to the provided email
                        send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendation)
                        st.success("Results sent via Email.")    

    elif pick == 'Upload':
        st.subheader("Upload Input")
        plantpic = st.file_uploader("Upload an image", ['jpg', 'png', 'gif', 'webp', 'tiff', 'psd', 'raw', 'bmp', 'jfif'], False, label_visibility="hidden")
        
        st.subheader("Select A Plant")
        select = st.selectbox("Select Plant", ('Lettuce', 'Cauliflower', 'Sugarcane', 'Pepper'), label_visibility="hidden")
        user_email = st.text_input("(Optional)Enter recipient email address", help="Enter the email address where you want to receive the result.")
        submit1 = st.button("Submit", use_container_width=True)
        
        if submit1:
            if not plantpic:
                st.write("Please upload an image!")
            else:
                # Perform prediction and display results
                st.subheader("Prediction: ")
                
                if select == 'Lettuce':
                    # Predict Lettuce disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model1, image_path, Lettuce_names)
                    pred1 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred1, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Lettuce " + predicted_class + ":"

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    st.subheader("Prediction Probabilities:")
                    generate_svm_plot(prediction, predicted_class)
    
                    recommendation = get_recommendations_as_string(predicted_class)

                    if not user_email == "":
                    # Call the function to send the result to the provided email
                        send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendation)
                        st.success("Results sent via Email.")
                        
                elif select == 'Cauliflower':
                    # Predict Cauliflower disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model2, image_path, Cauliflower_names)
                    pred2 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred2, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Cauliflower " + predicted_class + ":" 

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    st.subheader("Prediction Probabilities:")
                    generate_svm_plot(prediction, predicted_class)
    
                    recommendation = get_recommendations_as_string(predicted_class)

                    if not user_email == "":
                    # Call the function to send the result to the provided email
                        send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendation)
                        st.success("Results sent via Email.")
                        
                elif select == 'Sugarcane':
                    # Predict Sugarcane disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model3, image_path, Sugarcane_names)
                    pred3 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred3, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Sugarcane " + predicted_class + ":" 

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    st.subheader("Prediction Probabilities:")
                    generate_svm_plot(prediction, predicted_class)
    
                    recommendation = get_recommendations_as_string(predicted_class)

                    if not user_email == "":
                    # Call the function to send the result to the provided email
                        send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendation)
                        st.success("Results sent via Email.")
                        
                elif select == 'Pepper':
                    # Predict Pepper disease
                    image_path = plantpic
                    prediction, predicted_class = predict_disease(model4, image_path, Pepper_names)
                    pred4 = "Predicted Disease Class: " + predicted_class
                    st.image(plantpic, pred4, use_column_width=True)

                    # Generate and display recommendations
                    input_text = "Pepper " + predicted_class + ":" 

                    # Display specific recommendations for the predicted class
                    display_recommendations(predicted_class)
                    
                    # Generate SVM plot
                    st.subheader("Prediction Probabilities:")
                    generate_svm_plot(prediction, predicted_class)
    
                    recommendation = get_recommendations_as_string(predicted_class)

                    if not user_email == "":
                    # Call the function to send the result to the provided email
                        send_result_email(sender_email, sender_password, user_email, image_path, predicted_class, recommendation)
                        st.success("Results sent via Email.")        


# Define feedback categories
feedback_categories = ["User Experience", "Feature Requests", "Bug Reports", "General Comments"]

with tab3:
    st.title("Feedback")
    st.subheader("", divider='gray')
    
    st.subheader("Feedback Form")

    # Feedback text area
    feedback_text = st.text_area("Share your feedback with us...", help="Let us know how we can improve this app!")

    # Rating slider
    st.subheader("Rate Your Experience")
    rating = st.slider("Rate your experience (1-5)", 1, 5, help="Rate your overall experience using the app.")

    # Feedback category dropdown
    st.subheader("Select Feedback Category")
    feedback_category = st.selectbox("Select a feedback category", feedback_categories, help="Choose the category that best describes your feedback.")

    # Submit button
    submit_feedback = st.button("Submit Feedback")

    if submit_feedback:
        if feedback_text:
            # Submit feedback to database or log file
            st.success("Thank you for your feedback! We'll use it to improve the app.")
            # Send feedback via email
            send_feedback_email(sender_email, sender_password, recipient_email, feedback_text, rating, feedback_category)
        else:
            st.warning("Please provide feedback before submitting.")

with tab4:
    st.title("About Us")
    st.subheader("", divider='gray')
    
    st.write("""
        Welcome to the Crop Health Assessment App! Developed by a team of dedicated individuals passionate about technology and agriculture, our goal is to revolutionize the way farmers and agricultural enthusiasts assess and manage crop health.
             
        ### Our Mission

        At Crop Health Assessment, our mission is to empower farmers and crop enthusiasts with advanced machine learning technology to make informed decisions about crop health management. We aim to bridge the gap between cutting-edge technology and practical agricultural applications, ensuring that users can easily access and utilize the latest advancements in crop health assessment.

        ### Our Vision
        
        We envision a future where every farmer has access to advanced technology that simplifies crop health assessment and management. Through continuous innovation and collaboration with agricultural experts, we strive to create solutions that empower farmers, improve crop yields, and promote sustainable agriculture practices.
        
        ### Technologies Used

        The Crop Health Assessment App leverages a combination of modern technologies to deliver a seamless user experience and accurate crop health assessment:

        - Machine Learning: Our app utilizes machine learning models trained on extensive datasets of crop images to accurately identify and diagnose crop diseases.
        - Streamlit: We use Streamlit, a Python library, to develop interactive and user-friendly web applications for crop health assessment.
        - Keras and TensorFlow: We employ Keras, a high-level neural networks API, and TensorFlow, an open-source machine learning framework, for developing and training our deep learning models.
        - Python: The entire application is built using Python, a versatile and powerful programming language known for its simplicity and readability.
        - PIL (Python Imaging Library): PIL is used for image preprocessing tasks such as resizing and normalization, ensuring that input images are suitable for model prediction.
        - Matplotlib and NumPy: Matplotlib and NumPy are used for data visualization and numerical computations, respectively, enhancing the analysis and presentation of crop health assessment results.

        ### Get in Touch

        Have questions, feedback, or ideas for collaboration? We'd love to hear from you! Feel free to reach out to us at [cha.devteam223@gmail.com](mailto:cha.devteam223@gmail.com) and join us on our mission to revolutionize crop health assessment.

    """)

# Footer
st.markdown(
    """
    <style>
        .reportview-container .main footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

