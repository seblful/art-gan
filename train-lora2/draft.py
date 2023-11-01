import os
import json
from PIL import Image

HOME = os.getcwd()
IMAGES_PATH = os.path.join(HOME, "images")
JSON_PATH = os.path.join(IMAGES_PATH, 'metadata.jsonl')


def check_images():
    for file in os.listdir(os.path.join(os.getcwd(), 'images')):
        if file.endswith('.jpt'):
            image = Image.open(file)
            # image.show()


def extract_text_from_captions(folder_path, output_file):
    metadata = []

    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            # Read the caption text from the text file
            caption_file_path = os.path.join(folder_path, filename)
            with open(caption_file_path, 'r', encoding='utf-8') as file:
                caption_text = file.read().strip()

            # Get the corresponding image file name
            image_filename = os.path.splitext(filename)[0] + ".jpg"

            # Create a metadata entry
            metadata_entry = {
                "file_name": image_filename,
                "additional_feature": caption_text
            }

            metadata.append(metadata_entry)

    # Write the metadata to a JSONL file
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for entry in metadata:
            jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')


extract_text_from_captions(IMAGES_PATH, JSON_PATH)
