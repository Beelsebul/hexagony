from benchmark_agents import predict_move_hexhex
from benchmark_agents import hstic
import time

seq = "A1 B1"
print(hstic.predict_next_move(seq))

seq = "B1 A1"
print(predict_move_hexhex.predict_next_move(seq))

seq = "A1 B1"
print(predict_move_hexhex.predict_next_move(seq))
