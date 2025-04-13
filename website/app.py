from flask import Flask, request, render_template, jsonify
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import os

app = Flask(__name__)

# Define paths to model checkpoints and tokenizer
base_path = os.path.join(os.path.dirname(__file__), '..')
red_model_checkpoint = os.path.join(base_path, 'mini_blue_test', 'checkpoint-785460_ft_b_gen_5')
blue_model_checkpoint = os.path.join(base_path, 'mini_blue_test', 'checkpoint-785460_ft_b_gen_5')
tokenizer_path = os.path.join(base_path, 'tokenizer_2')

# Load the tokenizer and define special tokens
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
tokenizer.pad_token = '<|pad|>'
tokenizer.bos_token = '<|startoftext|>'
tokenizer.eos_token = '<|endoftext|>'
tokenizer.unk_token = '<|unk|>'

# If special tokens for agents are not in the vocabulary, you can add them:
# tokenizer.add_tokens(['<r>', '<b>'])

# Load models for red and blue agents
red_model = GPT2LMHeadModel.from_pretrained(red_model_checkpoint)
blue_model = GPT2LMHeadModel.from_pretrained(blue_model_checkpoint)

# Ensure the tokenizer embedding size matches the models
red_model.resize_token_embeddings(len(tokenizer))
blue_model.resize_token_embeddings(len(tokenizer))

def predict_next_tokens(model, input_text, agent_token, top_k=140):
    """
    Generates the next possible tokens based on the input text.
    
    Arguments:
      model: GPT2LMHeadModel used for generation.
      input_text: Current game move sequence (string).
      agent_token: Special agent token ('<r>' for red, '<b>' for blue).
      top_k: Number of top probable tokens to return.
    """
    # Construct the full input: <|startoftext|> + agent_token + input_text
    if not input_text:
        full_input = tokenizer.bos_token + agent_token
    else:
        full_input = tokenizer.bos_token + agent_token + input_text

    input_ids = tokenizer.encode(full_input, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids)

    next_token_logits = outputs.logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)

    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k)
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices[0])
    top_k_probs = top_k_probs[0].tolist()

    return top_k_tokens, top_k_probs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/next-move', methods=['POST'])
def next_move():
    data = request.get_json()
    moves = data.get('moves', '')

    # Determine whose turn it is based on the number of tokens in the sequence.
    # If even — it's red's turn, otherwise — blue's turn.
    current_turn = len(moves.split()) % 2
    if current_turn == 0:
        model = red_model
        agent_token = '<r>'
    else:
        model = blue_model
        agent_token = '<b>'

    top_moves, probs = predict_next_tokens(model, moves, agent_token, top_k=140)
    return jsonify({'next_moves': top_moves, 'probabilities': probs})

if __name__ == '__main__':
    app.run(debug=True)
