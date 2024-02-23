import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

"""
Approach inspired by "Clinical data based optimal STI strategies for HIV: a reinforcement learning approach", Ernst et al., 2006.
This implementation uses RandomForests instead of ExtraTrees.
"""

def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
        """
        The first step to perform FQI is to collect and store a dataset of samples.
        """
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    """
    Build the sequence of AVI Q-functions, learned using random forests.
    """
    nb_samples = S.shape[0]
    Qfunctions = []
    SA = np.append(S,A,axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter==0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = Qfunctions[-1].predict(S2A2)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2
        Q = RandomForestRegressor()
        Q.fit(SA,value)
        Qfunctions.append(Q)
    return Qfunctions