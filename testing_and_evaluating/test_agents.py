import numpy as np
import os
import re
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm
import pandas as pd
import csv
import string
import time
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from benchmark_agents import predict_move_hexhex
from benchmark_agents import hstic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_path = os.path.join(os.path.dirname(__file__), "..")
red_checkpoints_dir = os.path.join(base_path, "mini_red_test")
blue_checkpoints_dir = os.path.join(base_path, "mini_blue_test")
logs_dir = os.path.join(base_path, "logs")
tokenizer_path = os.path.join(base_path, "tokenizer")

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

columns = list(string.ascii_uppercase[:11])  # A-K
rows = [str(i) for i in range(1, 12)]         # 1-11
positions = [f"{col}{row}" for col in columns for row in rows]  # ['A1', 'A2', ..., 'K11']


def create_board():
    return np.zeros((11, 11), dtype=int)

def convert_move_to_coords(move):
    letter = move[0].upper()
    number = int(move[1:]) - 1
    x = ord(letter) - ord('A')
    y = number
    return x, y

def coords_to_move(x, y):
    letter = chr(x + ord('A'))
    number = y + 1
    return f"{letter}{number}"

def is_valid_move(move, size=11):
    if len(move) < 2:
        return False
    letter = move[0].upper()
    if not ('A' <= letter <= chr(ord('A') + size - 1)):
        return False
    try:
        number = int(move[1:])
        return 1 <= number <= size
    except ValueError:
        return False

def apply_moves(board, moves):
    applied_moves = []
    _allow_swap_rule = True
    for i, move in enumerate(moves):
        if not is_valid_move(move):
            return 6, board
        x, y = convert_move_to_coords(move)
        if i == 1 and _allow_swap_rule:
            if move == moves[0]:
                board[y, x] = 0
                y, x = x, y
                swap_move = coords_to_move(x, y)
                board[y, x] = 2
                applied_moves = [(x, y)]
            else:
                board[y, x] = 2
                applied_moves.append((x, y))
        else:
            if board[y, x] != 0 or (i >= 2 and (x, y) in applied_moves):
                return 5, board
            board[y, x] = 1 if i % 2 == 0 else 2
            applied_moves.append((x, y))
    return None, board

def parse_moves_sequence(sequence):
    return sequence.split()

def get_neighbors(i, j, size):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    return [(i + di, j + dj) for di, dj in directions if 0 <= i + di < size and 0 <= j + dj < size]

def dfs(board, start, player, goal_edge):
    size = len(board)
    stack = [start]
    visited = set()
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            i, j = current
            if goal_edge(i, j):
                return True
            for ni, nj in get_neighbors(i, j, size):
                if board[ni, nj] == player and (ni, nj) not in visited:
                    stack.append((ni, nj))
    return False

def check_winner_red(board):
    size = len(board)
    for j in range(size):
        if board[0, j] == 1:
            if dfs(board, (0, j), 1, lambda i, j: i == size - 1):
                return True
    return False

def check_winner_blue(board):
    size = len(board)
    for i in range(size):
        if board[i, 0] == 2:
            if dfs(board, (i, 0), 2, lambda i, j: j == size - 1):
                return True
    return False

def check_winner(board):
    if check_winner_red(board):
        return 3
    elif check_winner_blue(board):
        return 4
    else:
        return 2

def play_game(moves_sequence):
    board = create_board()
    moves = parse_moves_sequence(moves_sequence)
    error, board = apply_moves(board, moves)
    if error:
        return error
    winner = check_winner(board)
    return winner


def load_model(checkpoint_path):
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    return model

def predict_next_move(model, seq):
    input_text = '<|startoftext|> ' + seq if seq else '<|startoftext|>'
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 125, dim=-1)
    
    for i, (token_prob, token_id) in enumerate(zip(top_k_probs[0].tolist(), top_k_indices[0].tolist())):
        move_candidate = tokenizer.decode([token_id]).strip()
        if move_candidate in seq.split():
            continue
        temp_seq = seq + ' ' + move_candidate if seq else move_candidate
        if play_game(temp_seq) in [2, 3, 4]:

            return move_candidate, i  
    return None, None

def batch_predict_next_moves(model, sequences):
    input_texts = ['<|startoftext|> ' + seq if seq else '<|startoftext|>' for seq in sequences]
    inputs = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 125, dim=-1)
    
    results = []
    for batch_idx, seq in enumerate(sequences):
        chosen_move = None
        error_count = 0
        played = set(seq.split())
        for i in range(top_k_indices.size(1)):
            token_id = top_k_indices[batch_idx, i].item()
            move_candidate = tokenizer.decode([token_id]).strip()
            if move_candidate in played:
                continue
            temp_seq = seq + ' ' + move_candidate if seq else move_candidate
            if play_game(temp_seq) in [2, 3, 4]:
                chosen_move = move_candidate
                error_count = i 
                break
        results.append((chosen_move, error_count))
    return results

def play_games(model, playing_as, checkpoint_path, all_games_df, stats_df):
    openings_file = os.path.join(base_path, "openings.txt")
    with open(openings_file, 'r', encoding='utf-8') as f:
        openings = [line.strip() for line in f if line.strip()]
    
    games = []
    for opening in openings:
        moves = opening.split()
        status = play_game(opening)
        games.append({'seq': opening, 'status': status, 'moves': moves, 
                      'always_first': True, 'error_tokens': 0})
    
    while any(game['status'] == 2 for game in games):
        if playing_as == 'red':
            blue_turn_games = [g for g in games if g['status'] == 2 and len(g['moves']) % 2 == 1]
            for game in blue_turn_games:
                seq = ' '.join(game['moves'])
                next_move = hstic.predict_next_move(seq)
                if next_move:
                    temp_seq = game['seq'] + ' ' + next_move if game['seq'] else next_move
                    status = play_game(temp_seq)
                    if status in [2, 3, 4]:
                        game['moves'].append(next_move)
                        game['seq'] = temp_seq
                        game['status'] = status
                    else:
                        game['status'] = status
                else:
                    game['status'] = 5
            
            red_turn_games = [g for g in games if g['status'] == 2 and len(g['moves']) % 2 == 0]
            if red_turn_games:
                sequences = [' '.join(g['moves']) for g in red_turn_games]
                next_moves = batch_predict_next_moves(model, sequences)
                for game, (move, err_count) in zip(red_turn_games, next_moves):
                    if move:
                        temp_seq = game['seq'] + ' ' + move if game['seq'] else move
                        status = play_game(temp_seq)
                        if status in [2, 3, 4]:
                            game['moves'].append(move)
                            game['seq'] = temp_seq
                            game['status'] = status

                            if err_count > 0:
                                game['always_first'] = False
                            game['error_tokens'] += err_count
                        else:
                            game['status'] = status
                    else:
                        game['status'] = 5
        else: 

            red_turn_games = [g for g in games if g['status'] == 2 and len(g['moves']) % 2 == 0]
            for game in red_turn_games:
                seq = ' '.join(game['moves'])
                next_move = hstic.predict_next_move(seq)
                if next_move:
                    temp_seq = game['seq'] + ' ' + next_move if game['seq'] else next_move
                    status = play_game(temp_seq)
                    if status in [2, 3, 4]:
                        game['moves'].append(next_move)
                        game['seq'] = temp_seq
                        game['status'] = status
                    else:
                        game['status'] = status
                else:
                    game['status'] = 5
            
            blue_turn_games = [g for g in games if g['status'] == 2 and len(g['moves']) % 2 == 1]
            if blue_turn_games:
                sequences = [' '.join(g['moves']) for g in blue_turn_games]
                next_moves = batch_predict_next_moves(model, sequences)
                for game, (move, err_count) in zip(blue_turn_games, next_moves):
                    if move:
                        temp_seq = game['seq'] + ' ' + move if game['seq'] else move
                        status = play_game(temp_seq)
                        if status in [2, 3, 4]:
                            game['moves'].append(move)
                            game['seq'] = temp_seq
                            game['status'] = status
                            if err_count > 0:
                                game['always_first'] = False
                            game['error_tokens'] += err_count
                        else:
                            game['status'] = status
                    else:
                        game['status'] = 5
    
    for game in games:
        if game['status'] == 3:
            winner = 'Red' if playing_as == 'red' else 'Test Model'
        elif game['status'] == 4:
            winner = 'Blue' if playing_as == 'blue' else 'Test Model'
        else:
            winner = 'Invalid'
        all_games_df = pd.concat([all_games_df, pd.DataFrame([{
            'Checkpoint': checkpoint_path,
            'Opponent': 'test_model',
            'Game Moves': ' '.join(game['moves']),
            'Winner': winner
        }])], ignore_index=True)
        
        checkpoint_idx = stats_df.index[stats_df['Checkpoint'] == checkpoint_path].tolist()
        if not checkpoint_idx:
            stats_df = pd.concat([stats_df, pd.DataFrame([{
                'Checkpoint': checkpoint_path, 
                'Wins': 0, 
                'Losses': 0, 
                'AlwaysFirst': 0,  
                'ErrorTokens': 0 
            }])], ignore_index=True)
            checkpoint_idx = [len(stats_df) - 1]
        idx = checkpoint_idx[0]
        if (playing_as == 'red' and game['status'] == 3) or (playing_as == 'blue' and game['status'] == 4):
            stats_df.at[idx, 'Wins'] += 1
        elif (playing_as == 'red' and game['status'] == 4) or (playing_as == 'blue' and game['status'] == 3):
            stats_df.at[idx, 'Losses'] += 1
        if game['always_first']:
            stats_df.at[idx, 'AlwaysFirst'] += 1
        stats_df.at[idx, 'ErrorTokens'] += game['error_tokens']
    
    return all_games_df, stats_df

def main():
    all_games_df = pd.DataFrame(columns=['Checkpoint', 'Opponent', 'Game Moves', 'Winner'])
    stats_df = pd.DataFrame(columns=['Checkpoint', 'Wins', 'Losses', 'AlwaysFirst', 'ErrorTokens'])
    
    red_checkpoints = sorted([os.path.join(red_checkpoints_dir, cp) for cp in os.listdir(red_checkpoints_dir) if 'checkpoint-' in cp],
                             key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    blue_checkpoints = sorted([os.path.join(blue_checkpoints_dir, cp) for cp in os.listdir(blue_checkpoints_dir) if 'checkpoint-' in cp],
                              key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    
    total_checkpoints = len(red_checkpoints) + len(blue_checkpoints)
    with open(os.path.join(base_path, "openings.txt"), "r", encoding='utf-8') as f:
        openings = [line.strip() for line in f if line.strip()]
    total_games = total_checkpoints * len(openings)
    
    start_time = time.time()
    with tqdm(total=total_games, desc="Processing Games", unit="game") as pbar:

        for red_cp in red_checkpoints:
            model = load_model(red_cp)
            all_games_df, stats_df = play_games(model, 'red', red_cp, all_games_df, stats_df)
            pbar.update(len(openings))
            del model
            torch.cuda.empty_cache()
        
        for blue_cp in blue_checkpoints:
            model = load_model(blue_cp)
            all_games_df, stats_df = play_games(model, 'blue', blue_cp, all_games_df, stats_df)
            pbar.update(len(openings))
            del model
            torch.cuda.empty_cache()
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nВремя выполнения: {execution_time} сек")

    all_games_df.to_csv(os.path.join(logs_dir, 'generations_625_games.csv'), index=False, sep=';')
    stats_df.to_csv(os.path.join(logs_dir, 'generations_625.csv'), index=False, sep=';')

if __name__ == "__main__":
    main()
