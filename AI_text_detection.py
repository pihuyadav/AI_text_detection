{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b01f227a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Python Version:\n",
      "3.10.14 (main, Mar 21 2024, 11:24:58) [Clang 14.0.6 ]\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "print(\"Python Version:\")\n",
    "print(sys.version)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8bc9a036",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "from flask_cors import CORS\n",
    "import math\n",
    "import torch\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a0f97420",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import GPT2LMHeadModel, GPT2TokenizerFast\n",
    "\n",
    "model_id = \"openai-community/gpt2-large\"\n",
    "model = GPT2LMHeadModel.from_pretrained(model_id)\n",
    "tokenizer = GPT2TokenizerFast.from_pretrained(model_id)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cea7bce5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:8081\n",
      "\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "127.0.0.1 - - [25/Apr/2024 12:35:17] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:00<?, ?it/s]huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...\n",
      "To disable this warning, you can either:\n",
      "\t- Avoid using `tokenizers` before the fork if possible\n",
      "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n",
      "  0%|                                                     | 0/1 [00:09<?, ?it/s]\n",
      "127.0.0.1 - - [25/Apr/2024 12:35:27] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "19.06122589111328\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [25/Apr/2024 12:35:53] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "Token indices sequence length is longer than the specified maximum sequence length for this model (1505 > 1024). Running this sequence through the model will result in indexing errors\n",
      " 33%|███████████████                              | 1/3 [00:44<01:29, 44.59s/it]\n",
      "127.0.0.1 - - [25/Apr/2024 12:36:38] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "22.450927734375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [25/Apr/2024 12:37:34] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:04<?, ?it/s]\n",
      "127.0.0.1 - - [25/Apr/2024 12:37:39] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "12.309859275817871\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [25/Apr/2024 12:39:02] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:04<?, ?it/s]\n",
      "127.0.0.1 - - [25/Apr/2024 12:39:07] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "12.309859275817871\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [25/Apr/2024 12:39:26] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:02<?, ?it/s]\n",
      "127.0.0.1 - - [25/Apr/2024 12:39:29] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "12.309859275817871\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [25/Apr/2024 12:39:51] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/2 [00:15<?, ?it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "9.723694801330566\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [25/Apr/2024 12:40:07] \"POST /detect-text HTTP/1.1\" 200 -\n",
      "127.0.0.1 - - [25/Apr/2024 12:40:54] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:07<?, ?it/s]\n",
      "127.0.0.1 - - [25/Apr/2024 12:41:02] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "13.238763809204102\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [25/Apr/2024 12:41:32] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:06<?, ?it/s]\n",
      "127.0.0.1 - - [25/Apr/2024 12:41:39] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "21.29853630065918\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [25/Apr/2024 12:45:19] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:19<?, ?it/s]\n",
      "127.0.0.1 - - [25/Apr/2024 12:45:39] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10.004142761230469\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [27/Apr/2024 16:47:32] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:29<?, ?it/s]\n",
      "127.0.0.1 - - [27/Apr/2024 16:48:03] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10.004142761230469\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [27/Apr/2024 16:50:32] \"OPTIONS /detect-text HTTP/1.1\" 200 -\n",
      "  0%|                                                     | 0/1 [00:19<?, ?it/s]\n",
      "127.0.0.1 - - [27/Apr/2024 16:50:51] \"POST /detect-text HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "11.456793785095215\n"
     ]
    }
   ],
   "source": [
    "app = Flask(__name__)\n",
    "CORS(app)\n",
    "@app.route('/detect-text', methods=['POST'])\n",
    "def detect_text():\n",
    "    # Get the text data from the request\n",
    "    try:\n",
    "        data = request.get_json()\n",
    "        text = data['text'] \n",
    "        if(len(text)<300):\n",
    "            return jsonify({\"perplexity\": \"Need a minimum of 300 characters to provide accurate analysis\"})\n",
    "        encodings = tokenizer(text, return_tensors=\"pt\")\n",
    "        max_length = model.config.n_positions #the maximum number of tokens the model can handle in one pass\n",
    "        stride = 512\n",
    "        seq_len = encodings.input_ids.size(1) #after tokenization the full length of the text using sub-word tokenization \n",
    "        nlls = []\n",
    "        prev_end_loc = 0\n",
    "        for begin_loc in tqdm(range(0, seq_len, stride)):\n",
    "            end_loc = min(begin_loc + max_length, seq_len)\n",
    "            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop\n",
    "            input_ids = encodings.input_ids[:, begin_loc:end_loc]\n",
    "            target_ids = input_ids.clone()\n",
    "            target_ids[:, :-trg_len] = -100\n",
    "                  \n",
    "            with torch.no_grad():\n",
    "                outputs = model(input_ids, labels=target_ids)\n",
    "                # loss is calculated using CrossEntropyLoss which averages over valid labels\n",
    "                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels\n",
    "                # to the left by 1.\n",
    "                neg_log_likelihood = outputs.loss\n",
    "        \n",
    "            nlls.append(neg_log_likelihood)\n",
    "            prev_end_loc = end_loc\n",
    "            if end_loc == seq_len:\n",
    "                break\n",
    "        ppl = torch.exp(torch.stack(nlls).mean())\n",
    "        answer=ppl.item()\n",
    "        print(answer)\n",
    "        # Return the detected text as a JSON response\n",
    "        return jsonify({\"perplexity\": answer})\n",
    "    except Exception as e:\n",
    "        return jsonify({'error': str(e)}), 500\n",
    "\n",
    "    \n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(port=8081)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0bc8b0fe",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
