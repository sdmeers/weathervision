import requests
import base64
import json
import argparse

def classify_image(image_path):
    """
    Sends an image to the Ollama API for weather classification.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The weather classification.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    data = {
        #"model": "llama3.2-vision",
        "model": "gemma3:4b-it-qat",
        "prompt": "You are a weather classification expert. Your task is to analyze images and classify the weather conditions into one of the following categories: \"Sunny\", \"Cloudy\", \"Rainy\", \"Snowy\", \"Sunrise\", or \"Foggy\". Respond with only a single word corresponding to the most prominent weather condition in the image.",
        "stream": False,
        "images": [encoded_string]
    }

    response = requests.post("http://localhost:11434/api/generate", json=data)

    if response.status_code == 200:
        return response.json()['response'].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify weather conditions in an image using Ollama.')
    parser.add_argument('--image', type=str, required=True, help='The path to the image file.')
    args = parser.parse_args()

    classification = classify_image(args.image)
    print(f"The predicted weather is: {classification}")

