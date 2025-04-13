#!/usr/bin/env python3
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from hexhex.utils.utils import load_model
from hexhex.logic.hexboard import Board
from hexhex.logic.hexgame import MultiHexGame

model_path = os.path.join(script_dir, "models", "11_2w4_2000.pt")
model = load_model(model_path)

def parse_move(move_str):
    letter, number = move_str[0], move_str[1:]
    y = ord(letter.upper()) - ord('A')
    x = int(number) - 1
    return (x, y)

def format_move(move_tuple):
    x, y = move_tuple
    return f"{chr(65 + y)}{x + 1}"

def predict_next_move(moves_sequence_str):
    board = Board(size=model.board_size, switch_allowed=False)
    for m in moves_sequence_str.split():
        board.set_stone(parse_move(m))
    game = MultiHexGame(
        boards=[board],
        models=[model],
        noise=None,
        noise_parameters=None,
        temperature=0.1,
        temperature_decay=1.0
    )
    game.batched_single_move(model)
    return format_move(board.move_history[-1][1])
