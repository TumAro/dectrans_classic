from experiments.zero_force import zeroForce
from experiments.rollout_random import randomForce
from experiments.heuristic import heuristicForce
from experiments.inspect_data import inspect_data
from src.dataset import make_seq_and_save

if __name__ == "__main__":
    make_seq_and_save()
    inspect_data()
