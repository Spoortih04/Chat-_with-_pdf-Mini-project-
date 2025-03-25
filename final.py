import os
import re
import spacy
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from urllib.parse import urljoin
import face_recognition
from deepface import DeepFace
import uuid
import csv

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Folder paths
dataset_folder = "E:/7th SEM/final/input_images"  # Folder containing known criminal images
temp_folder = "temp_images"  # Temporary folder to save images from the article
os.makedirs(temp_folder, exist_ok=True)

# Load crime-related keywords from a CSV or text file
def load_crime_keywords(file_path):
    crime_keywords = []
    if file_path.endswith('.csv'):
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header if CSV has one
            for row in reader:
                crime_keywords.append(row[0].strip())
    elif file_path.endswith('.txt'):
        with open(file_path, mode='r') as file:
            crime_keywords = [line.strip() for line in file.readlines()]
    else:
        print("Unsupported file format.")
    return crime_keywords

# Read keywords from a CSV or text file
crime_keywords = load_crime_keywords('crime_keywords.csv')  # Path to your keyword file

# Improved Text Classification with the loaded keyword list
def classify_text(text):
    pattern = re.compile(r'\b(?:' + '|'.join(crime_keywords) + r')\b', re.IGNORECASE)
    if pattern.search(text):
        return "criminal"
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"] and any(keyword in ent.text.lower() for keyword in crime_keywords):
            return "criminal"
    return "not criminal"

# Extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

# Encode dataset images
def encode_dataset(dataset_folder):
    encodings = []
    names = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    for img_name in os.listdir(dataset_folder):
        if not img_name.lower().endswith(valid_extensions):
            continue
        img_path = os.path.join(dataset_folder, img_name)
        try:
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                encodings.append(face_encodings[0])
                names.append(img_name)
        except Exception as e:
            print(f"Error processing file {img_name}: {e}")
    return encodings, names

# Function to analyze images and find a match
def find_culprit_in_images(dataset_encodings, dataset_names, image_paths):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    for img_path in image_paths:
        if not img_path.lower().endswith(valid_extensions):
            continue
        try:
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image)
            if not face_encodings:
                continue
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(dataset_encodings, face_encoding)
                if any(matches):
                    match_index = matches.index(True)
                    culprit_name = dataset_names[match_index]
                    return f"Culprit found: {culprit_name}"
        except Exception as e:
            print(f"Error processing file {img_path}: {e}")
    return "Culprit not found."

# Extract images from URL
def extract_images_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        images = soup.find_all("img")
        base_url = url
        image_urls = [urljoin(base_url, img["src"]) for img in images if "src" in img.attrs]
        return image_urls
    except Exception as e:
        print(f"Error fetching images from URL: {e}")
        return []

# Download images and save them to a temporary folder
def download_images(image_urls, temp_folder):
    image_paths = []
    for idx, img_url in enumerate(image_urls):
        try:
            response = requests.get(img_url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_path = os.path.join(temp_folder, f"image_{uuid.uuid4().hex}.jpg")
                img.save(img_path)
                image_paths.append(img_path)
        except Exception as e:
            print(f"Error downloading image {img_url}: {e}")
    return image_paths

# Analyze images for emotional traits using DeepFace
def analyze_images_for_behavior(image_paths):
    analysis_results = []
    for img_path in image_paths:
        try:
            analysis = DeepFace.analyze(img_path, actions=["emotion"], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]
            dominant_emotion = analysis.get("dominant_emotion", "unknown")
            if dominant_emotion in ["angry", "disgust", "fear"]:
                analysis_results.append((img_path, dominant_emotion))
        except Exception as e:
            print(f"Error analyzing image {img_path}: {e}")
    return analysis_results

# Identify and display the criminal based on behavioral analysis
def identify_and_display_criminal(analysis_results, dataset_folder):
    if analysis_results:
        suspect = analysis_results[0]
        print(f"Potential criminal identified: Emotion - {suspect[1]}")
        suspect_image = Image.open(suspect[0])
        suspect_image.show()
        # Save the suspect image to the dataset folder
        suspect_image_path = os.path.join(dataset_folder, f"suspect_{uuid.uuid4().hex}.jpg")
        suspect_image.save(suspect_image_path)
        print(f"Suspect image saved to dataset: {suspect_image_path}")
    else:
        print("No potential criminal identified. Ensure the images contain clear faces and emotions.")

# Main function to handle user input and processing
def main():
    url = input("Enter the URL of the article: ")
    print("Extracting text from the article...")
    text = extract_text_from_url(url)
    if not text:
        print("Could not extract text from the provided URL. Please try another.")
        return

    print("Classifying the article...")
    classification = classify_text(text)

    if classification == "criminal":
        print("The article is classified as related to crime.")
        print("Extracting images from the article...")
        image_urls = extract_images_from_url(url)
        if not image_urls:
            print("No images found in the article.")
            return

        print("Downloading images...")
        image_paths = download_images(image_urls, temp_folder)
        if not image_paths:
            print("No valid images could be downloaded.")
            return

        print("Encoding dataset images...")
        dataset_encodings, dataset_names = encode_dataset(dataset_folder)

        print("Finding culprit in the images...")
        culprit_message = find_culprit_in_images(dataset_encodings, dataset_names, image_paths)
        if culprit_message != "Culprit not found.":
            print(culprit_message)
        else:
            print("No culprit found in the images. Proceeding to behavioral analysis...")
            print("Analyzing images for emotional traits...")
            analysis_results = analyze_images_for_behavior(image_paths)
            identify_and_display_criminal(analysis_results, dataset_folder)

        for img_path in image_paths:
            os.remove(img_path)

    else:
        print("The article is not related to crime.")

if __name__ == "__main__":
    main()