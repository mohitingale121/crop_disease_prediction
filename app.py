import mysql.connector
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
from datetime import datetime

# Initialize Flask app
app = Flask(_name_)
app.secret_key = 'your_secret_key'  # Set a secret key for sessions

# MySQL Database Configuration
MYSQL_HOST = 'localhost'  # Or your MySQL host
MYSQL_DATABASE = 'plant_disease_db'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'root'

# MySQL Connection
def get_db_connection():
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        database=MYSQL_DATABASE,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    return connection

# Paths for model, disease info, and supplement info
MODEL_PATH = "trained_plant_disease_model.keras"
DISEASE_INFO_PATH = "disease_info.csv"
SUPPLEMENT_INFO_PATH = "supplement_info.csv"

# Load model and data
model = tf.keras.models.load_model(MODEL_PATH)
disease_info = pd.read_csv(DISEASE_INFO_PATH, encoding='ISO-8859-1')
supplement_info = pd.read_csv(SUPPLEMENT_INFO_PATH)

# Define class names (same as before)
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry__Powdery_mildew',
    'Cherry__healthy', 'Corn__Cercospora_leaf_spot Gray_leaf_spot',
    'Corn__Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn__healthy',
    'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy',
    'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
    'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
    'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot',
    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Predict the disease based on the image
def model_prediction(test_image):
    from tensorflow.keras.preprocessing import image

    img = image.load_img(test_image, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert image to batch of size 1
    prediction = model.predict(img_array)
    return np.argmax(prediction)

# Save registration and prediction data to MySQL
def save_registration_data(form_data, disease_name, city_name):
    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'}), 500

    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO registrations (name, email, city, gender, country, message, disease_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        data = (form_data['name'], form_data['email'], city_name, form_data['gender'], form_data['country'], form_data['message'], disease_name)
        cursor.execute(query, data)
        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error inserting into database: {err}")
        return jsonify({'error': 'Failed to save data'}), 500
    finally:
        cursor.close()
        connection.close()

# Get disease info from the dataset
def get_disease_info(disease_name):
    disease_rows = disease_info[disease_info['disease_name'] == disease_name]
    if not disease_rows.empty:
        disease_description = disease_rows['description'].values[0]
        possible_steps = disease_rows['Possible Steps'].values[0]
        image_url = disease_rows['image_url'].values[0]
    else:
        disease_description = "Description not available."
        possible_steps = "No steps available."
        image_url = "No image available."
    
    supplement_details = supplement_info[supplement_info['Disease'] == disease_name]
    return disease_description, possible_steps, image_url, supplement_details

# Ensure the uploads directory exists
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@app.route('/')
def home():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    city = request.form.get('city')
    gender = request.form.get('gender')
    country = request.form.get('country')
    message = request.form.get('message')

    # Validate form data
    if not name or not email or not city or not gender or not country:
        return redirect(url_for('home', error="All fields (name, email, city, gender, country) are required"))
    
    form_data = {'name': name, 'email': email, 'city': city, 'gender': gender, 'country': country, 'message': message}
    session['form_data'] = form_data  # Store form data in session
    return redirect(url_for('index'))  # Redirect without form_data in URL

@app.route('/index')
def index():
    form_data = session.get('form_data')  # Retrieve form data from session
    return render_template('index.html', form_data=form_data)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    # Predict the disease
    result_index = model_prediction(file_path)
    disease_name = class_names[result_index]

    # Save the registration form data and prediction results to MySQL
    form_data = session.get('form_data')  # Capture form data (such as name, email, city)
    city_name = form_data['city']
    save_registration_data(form_data, disease_name, city_name)  # Save to MySQL without image_filename

    # Get disease details
    disease_description, possible_steps, image_url, supplement_details = get_disease_info(disease_name)

    # Prepare supplement information
    supplements = []
    for _, supplement in supplement_details.iterrows():
        supplement_info = {
            'name': supplement.get('supplement_name', 'Name not available'),
            'image': supplement.get('supplement image', None),
            'link': supplement.get('buy link', '#')
        }
        supplements.append(supplement_info)
    
    return render_template('result.html', disease_name=disease_name, description=disease_description, steps=possible_steps, image_url=image_url, supplements=supplements)

if _name_ == "_main_":
    app.run(debug=True)