
from flask import Flask, request, jsonify
from flask_cors import CORS
import math
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

print('here')
model_id = "openai-community/gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

app = Flask(__name__)
CORS(app)
@app.route('/node')
def nindex():
    return 'INDEX'
@app.route('/detect-text', methods=['POST'])
def detect_text():
    # Get the text data from the request
    try:
        data = request.get_json()
        text = data['text'] 
        if(len(text)<300):
            return jsonify({"perplexity": "Need a minimum of 300 characters to provide accurate analysis"})
        encodings = tokenizer(text, return_tensors="pt")
        max_length = model.config.n_positions #the maximum number of tokens the model can handle in one pass
        stride = 512
        seq_len = encodings.input_ids.size(1) #after tokenization the full length of the text using sub-word tokenization 
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
                  
            outputs = model(input_ids, labels=target_ids)
                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
            neg_log_likelihood = outputs.loss
        
            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        avg_nll = sum(nlls) / len(nlls)
        # Calculate perplexity as exponentiation of the average negative log likelihood
        ppl = math.exp(avg_nll)
        print(ppl)
        # Return the detected text as a JSON response
        return jsonify({"perplexity": answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    

if __name__ == '__main__':
    app.run()
