import math
import numpy as np
import random as rand

actions = [[0,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,1,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1],[1,0,1,1,1,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0]]

def determineRegrets_Theoretical(actions):
    theoretical_LR = round(math.sqrt(math.log1p( len(actions) / len(actions[0]) ) ),4) 
    regret = round(2*theoretical_LR,4)
    n_actions = len(actions)
    n_rounds = len(actions[0])
    print(f'With {n_actions} actions (W/L)and {n_rounds} rounds(games), the Theoretical learning rate = {theoretical_LR},Expected Regret = {regret} for both FTPL EW')
    return theoretical_LR, regret

def FTPL (actions,n_rounds,LR):
    payoff = 0
    #hallucination is h*(number of times in a row we get tails on an epsilon bias coin ... tails bias = 1-epsilon, h=1)
    tails_bias = 1-LR
    hallucinations =[tails_in_a_row(tails_bias) for j in range(len(actions))]
    idx= getMaxIndex(hallucinations)
    sums = [hallucinations[a] for a in range(len(actions))]
    for round_idx in range(n_rounds):
        if round_idx == 0:
            #choose the action w the highest hallucination for round 1
            pick = actions[idx][round_idx]
        else:
            for i,action in enumerate(actions):
                sums[i] +=action[round_idx-1]
            pick_idx = getMaxIndex(sums)
            pick = actions[pick_idx][round_idx]
        payoff += pick
    print(f"FTPL payoff = {payoff}")
    return payoff

def tails_in_a_row(tails_bias):
    how_many=0
    while True:
        if rand.random() <= tails_bias:
            how_many+=1
        else:
            return how_many

def getMaxIndex(arr):
    max = 0
    for idx,el in enumerate(arr):
        curr_max= arr[max]
        if el>curr_max:
            max= idx
    return max


#######################################################
def EW(actions,n_rounds, LR):
    payoff = 0
    probabilities=[0 for i in range(len(actions))]
    idx = rand.randint(0,len(actions)-1)
    for round_idx in range(n_rounds):
        if round_idx == 0:
            pick = actions[idx][round_idx]
        else:
            for i,action in enumerate(actions):
                top = math.pow(1+LR, sum(action[:round_idx]))
                bottom = getSumAllOtherActionProbs(actions,round_idx,LR)
                action_prob = round(top/bottom,4)
                probabilities[i] = action_prob
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            pick_idx = np.random.choice([i for i in range(len(actions))],p=probabilities)
            pick = actions[pick_idx][round_idx]
        payoff+=pick
    print(f"EW payoff = {payoff}")
    return payoff

#calculate (1+e)^Vj^i of every other action and sum this for the bottom of the ew probability equation
def getSumAllOtherActionProbs(arr,round_idx,LR):
    action_probs = []
    for i,each_arr in enumerate(arr):
        curr_action_prob = math.pow(1+LR, sum(each_arr[:round_idx]))
        action_probs.append(curr_action_prob)
    sum_action_probs = sum(action_probs)
    return sum_action_probs



#####################################################
def OPT(actions):
    sums = [0 for i in range(len(actions))]
    for idx,action in enumerate(actions):
        sums[idx] = sum(action)
    max_action_index = getMaxIndex(sums)
    print(f"OPT Payoff = {sums[max_action_index]}")
    return sums[max_action_index]

def regret(ftpl_payoff,ew_payoff,OPT_payoff,n):
    ftpl_regret = (1/n)*(OPT_payoff - ftpl_payoff)
    ew_regret = (1/n)*(OPT_payoff - ew_payoff)
    print(f"Actual FTPL Regret = {ftpl_regret}. Actual EW Regret = {ew_regret}")
    return ftpl_regret,ew_regret


total_numb_games = len(actions[0])
variable_number_of_rounds = [round(total_numb_games/4),round(total_numb_games/3),round(total_numb_games/2),total_numb_games]

for numb_games in variable_number_of_rounds:
    actions = [[0,1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,1,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1],[1,0,1,1,1,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0]]
    print(f"\nBulls first {numb_games} games")
    actions[0] = actions[0][:numb_games]
    actions[1] = actions[1][:numb_games]
    theo_LR,exp_regret = determineRegrets_Theoretical(actions)
    FTPL_pay = FTPL(actions,numb_games,theo_LR)
    EW_pay = EW(actions,numb_games,theo_LR)
    OPT_pay = OPT(actions)
    ftpl_regret,ew_regret = regret(FTPL_pay,EW_pay,OPT_pay,numb_games)