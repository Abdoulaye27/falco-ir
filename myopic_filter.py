import numpy as np

def bayes_filter(confidence_score):
    # Reward Matrix
    Reward = np.array([[20, -100], [-100, 100], [0, -10]])

    R_NP_DA = Reward[0][0]
    R_P_DA = Reward[0][1]
    R_NP_A = Reward[1][0]
    R_P_A = Reward[1][1]
    R_NP_G = Reward[2][0]
    R_P_G = Reward[2][1]

    #confidence_score = 0.45  # Object-detection algorithm
    Pr_P = 0.5  # Probability of finding a person based on mission scenario
    Pr_BB = 0.4  # Probability of observing a bounding box

    Pr_P_BB = (confidence_score * Pr_P) / Pr_BB  # Probability that there is a person present given the observation of a bounding box
    Pr_NP_BB = 1 - Pr_P_BB  # Probability that there is no person present given the observation of a bounding box

    DA = Pr_NP_BB * R_NP_DA + Pr_P_BB * R_P_DA  # Continue mission reward model
    A = Pr_NP_BB * R_NP_A + Pr_P_BB * R_P_A  # Alert model reward model
    G = Pr_NP_BB * R_NP_G + Pr_P_BB * R_P_G  # Gather information reward model 

    '''
    print("Alert reward is ", A)
    print("Gather info reward is ", G)
    print("Continue mission is ", DA)
    '''

    Decision = max(A, DA, G)

    if Decision == A:
        action = 1 
        #print("ALERT!")
    if Decision == DA:
        action = 3 
        #print("CONTINUE!")
    if Decision == G:
        action = 2 
        #print("Gather Info!")

    return action

