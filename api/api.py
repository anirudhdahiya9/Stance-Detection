import argparse
from flask import Flask, request
import os
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
import torch


app = Flask(__name__)
app.config['DEBUG'] = True


label_list = ['unrelated', 'discuss', 'agree', 'disagree']
label_map = dict(zip(range(4), label_list))

def infer_model(headline, article):
    batch_input = tokenizer.batch_encode_plus([(headline, article)], pad_to_max_length=True,
                                              max_len=500, return_tensors='pt')
    for key, val in batch_input.items():
        batch_input[key] = val.to(device)
    model_outputs = model(**batch_input)
    predictions = [label_map[key] for key in model_outputs.argmax(dim=1).tolist()]
    return predictions


@app.route('/', methods=['GET'])
def home():
    return '<h1>This is a prototype api for the stance detection api.</h1>'

@app.route('/infer', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        if data:
            try:
                headline, article = data['headline'], data['article']
            except KeyError:
                missing_keys = {'headline', 'article'}.intersection(set(data.keys()))
                return f"Missing keys {missing_keys}.", 400

            pred_label = infer_model(headline, article)
            return str(pred_label), 200

        else:
            return "No data received", 400



if __name__=="__main__":
    parser = argparse.ArgumentParser(prog='Stance Detection API', description='API to expose stance detection'
                                                                              ' transformer model.')
    parser.add_argument('--model_path', default='../models/stop_jaldi', type=str)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    device = torch.device("cuda") if (args.cuda and torch.cuda.is_available()) else torch.device('cpu')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='../models/roberta/')
    model_path = args.model_path
    config = RobertaConfig.from_pretrained(model_path, num_labels=4)
    model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
    model.to(device)


    app.run(host='0.0.0.0', port='2347')
