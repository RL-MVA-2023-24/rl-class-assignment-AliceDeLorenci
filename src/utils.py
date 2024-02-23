
import numpy as np

def greedy_action(Q,s,nb_actions):
    """
    Return the greedy action with respect to the Q-function Q and the state s.
    """
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)