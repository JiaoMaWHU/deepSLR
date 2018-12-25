import numpy as np
from hmmlearn import hmm
import LoadData as DATA


def initmodel(num_state):
    load = DATA.LoadData("data")
    data = load.getdata()
    model=hmm.MultinomialHMM(n_components=num_state, n_iter=20, tol=0.01)


if __name__ == '__main__':
    initmodel(402)