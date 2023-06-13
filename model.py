from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

app = Flask(__name__, template_folder='.')
CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        image = Image.open(image)
    elif 'link' in request.form:
        link = request.form['link']
        image = Image.open(requests.get(link, stream=True).raw).convert('RGB')
    
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    response = {
        'caption': caption
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)