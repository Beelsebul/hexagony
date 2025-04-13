import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import os

# Specifying paths (измените при необходимости)
base_dir = os.path.join(os.path.dirname(__file__), '..')
model_checkpoint = os.path.join(base_dir, 'mini_red_test', 'checkpoint-1551420')
tokenizer_path = os.path.join(base_dir, "tokenizer_2")

# Loading tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

# Ensuring special tokens are set
tokenizer.pad_token = '<|pad|>'
tokenizer.bos_token = '<|startoftext|>'
tokenizer.eos_token = '<|endoftext|>'
tokenizer.unk_token = '<|unk|>'

# Updating model embeddings to match the tokenizer
model.resize_token_embeddings(len(tokenizer))

# Setting special tokens in model configuration
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.unk_token_id = tokenizer.unk_token_id

def predict_next_tokens(input_text, top_k=5):
    """
    Функция получает top-k вероятных следующих токенов.
    Если input_text пустой, используется только токен начала текста.
    Иначе к тексту добавляется префикс '<|startoftext|>'.
    """
    if not input_text:
        input_ids = torch.tensor([[tokenizer.bos_token_id]])
    else:
        input_text = '<|startoftext|>' + input_text
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(input_ids)
    
    next_token_logits = outputs.logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    
    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k)
    
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices[0])
    top_k_probs = top_k_probs[0].tolist()
    
    return top_k_tokens, top_k_probs

def generate_openings():
    """
    Генерирует варианты открытий по схеме:
    - 1-й ход: 10 вариантов,
    - 2-й ход: 2 варианта,
    - 3-й ход: 5 вариантов,
    - 4-й ход: 5 вариантов.
    
    Всего получится 10 * 2 * 5 * 5 = 500 открытий.
    Результат сохраняется в файл cheopenings.txt в текущей папке.
    """
    openings = []

    # Первый ход: 10 вариантов
    first_moves, _ = predict_next_tokens("", top_k=10)
    for move1 in first_moves:
        seq1 = move1.strip()
        # Второй ход: 2 варианта
        second_moves, _ = predict_next_tokens(seq1, top_k=2)
        for move2 in second_moves:
            seq2 = seq1 + " " + move2.strip()
            # Третий ход: 5 вариантов
            third_moves, _ = predict_next_tokens(seq2, top_k=5)
            for move3 in third_moves:
                seq3 = seq2 + " " + move3.strip()
                # Четвёртый ход: 5 вариантов
                fourth_moves, _ = predict_next_tokens(seq3, top_k=5)
                for move4 in fourth_moves:
                    opening = seq3 + " " + move4.strip()
                    openings.append(opening)

    # Определяем путь для файла cheopenings.txt (в той же папке, что и скрипт)
    output_path = os.path.join(os.path.dirname(__file__), "cheopenings.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for opening in openings:
            f.write(opening + "\n")
    
    print(f"Сохранено {len(openings)} открытий в {output_path}")

if __name__ == "__main__":
    generate_openings()
