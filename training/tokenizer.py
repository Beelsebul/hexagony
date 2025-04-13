import os
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

def split_move_tokens(line):
    # No longer needed to split A11 into A and 11, return the string as-is.
    return line

def get_tokenizer(file_path):
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Whitespace is now sufficient, as moves are no longer split
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    special_tokens = [
        '<|startoftext|>', '<|endoftext|>', '<|unk|>', '<|pad|>',
        '<r>', '<b>'
    ]
    trainer = trainers.BpeTrainer(special_tokens=special_tokens)

    # Preprocess text: leave lines as-is
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [split_move_tokens(line.strip()) for line in f if line.strip()]

    tokenizer.train_from_iterator(lines, trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token='<|startoftext|>',
        eos_token='<|endoftext|>',
        unk_token='<|unk|>',
        pad_token='<|pad|>'
    )
    fast.add_special_tokens({'additional_special_tokens': ['<r>', '<b>']})
    return fast

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_file = os.path.join(base_dir, 'hexhex_games_marked.txt')
    tokenizer_dir = os.path.join(base_dir, 'tokenizer_2')
    os.makedirs(tokenizer_dir, exist_ok=True)

    tokenizer = get_tokenizer(data_file)
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"✅ Tokenizer saved to {tokenizer_dir}")

    # === VALIDATION ===
    test_text = "<r> A11 B2 <b> K11"
    test_text_split = split_move_tokens(test_text)
    encoded = tokenizer.encode(test_text_split)
    decoded = tokenizer.decode(encoded)

    print("\n🔍 Tokenization check:")
    print("Original :", test_text)
    print("Processed:", test_text_split)
    print("Tokens   :", tokenizer.tokenize(test_text_split))
    print("Token IDs:", encoded)
    print("Decoded  :", decoded)
