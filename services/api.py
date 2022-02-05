from flask_cors import CORS
from flask import Flask, jsonify, request

from nlpmodule.tools.Preprocessing import text_preprocessing
from nlpmodule.tools.Classifier import classifier_treatment_load



app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Bienvenido a API KT-BERT"


@app.route("/text", methods=['POST'])
def predict_text():
    json = request.get_json(force=True)
    text = json['text']
    predict = classifier_treatment_load(text)
    
    result = {
         'text' : text
        ,'textprepros': text_preprocessing(text)
        ,'label': predict[0]
        ,'prob' : predict[1]
    }

    return result
