import random as rand
import math
import numpy as np

# generates random [0,1] data (according to specified number of rounds) for each action (according to specified number of actions). 
# Stores in all_actions, where action1 is at index 0, action2 and index 1 etc ...
# Probability of each action is stored at the corresponding index in probabilities list (action1 = all_actions[0], pr of action1 = probabilities[0])

def generateData(n_actions,n_rounds):
    all_actions=[]
    for action in range(n_actions):
        action_x = [rand.randint(0,1) for i in range(n_rounds)]
        all_actions.append(action_x)
    return all_actions

def determineRegrets_Theoretical(actions):
    theoretical_LR = round(math.sqrt(math.log1p( len(actions) / len(actions[0]) ) ),4) 
    regret = round(2*theoretical_LR,4)
    n_actions = len(actions)
    n_rounds = len(actions[0])
    # print(f'With {n_actions} actions and {n_rounds} rounds, the Theoretical learning rate = {theoretical_LR},Expected Regret = {regret} for both FTPL EW')
    return theoretical_LR, regret



###########################################################
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
    # print(f"FTPL payoff = {payoff}")
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
    # print(f"EW payoff = {payoff}")
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
    # print(f"OPT Payoff = {sums[max_action_index]}")
    return sums[max_action_index]

def regret(ftpl_payoff,ew_payoff,OPT_payoff,n):
    ftpl_regret = (1/n)*(OPT_payoff - ftpl_payoff)
    ew_regret = (1/n)*(OPT_payoff - ew_payoff)
    # print(f"Actual FTPL Regret = {ftpl_regret}. Actual EW Regret = {ew_regret}")
    return ftpl_regret,ew_regret


n_rounds= range(1,50)
regrets_fewRounds = []
regrets_manyRounds = []
LR_fewRounds=[]
LR_manyRounds =[]

analyzeFTPL=[]
analyzeEW=[]

for i in range(2,5):
    for n in n_rounds:
        actions = generateData(i,n)
        theo_LR,exp_regret = determineRegrets_Theoretical(actions)
        FTPL_pay = FTPL(actions,n,theo_LR)
        EW_pay = EW(actions,n,theo_LR)
        OPT_pay = OPT(actions)
        ftpl_regret,ew_regret = regret(FTPL_pay,EW_pay,OPT_pay,n)
        if n <20:
            regrets_fewRounds.append([ftpl_regret,ew_regret])
            LR_fewRounds.append(theo_LR)
        else:
            regrets_manyRounds.append([ftpl_regret,ew_regret])
            LR_manyRounds.append(theo_LR)
            analyzeFTPL.append([ftpl_regret,theo_LR,n,i,exp_regret])
            analyzeEW.append([ew_regret,theo_LR,n,i,exp_regret])
        # print("\n")
regrets_fewRounds = np.array(regrets_fewRounds)
regrets_manyRounds = np.array(regrets_manyRounds)

min_reg_idx_ftpl = 0
for j,data in enumerate(analyzeFTPL):
    if analyzeFTPL[j][0] < analyzeFTPL[min_reg_idx_ftpl][0]:
        min_reg_idx_ftpl = j
min_reg_idx_ew = 0
for k,data in enumerate(analyzeEW):
    if analyzeEW[k][0] < analyzeEW[min_reg_idx_ew][0]:
        min_reg_idx_ew = k
    


many_actions_fewRounds =[]
many_actions_manyRounds = []
LR_fewRoundsManyA=[]
LR_manyRoundsManyA =[]
for i in range(5,50):
    for n in n_rounds:
        actions = generateData(i,n)
        theo_LR,exp_regret = determineRegrets_Theoretical(actions)
        FTPL_pay = FTPL(actions,n,theo_LR)
        EW_pay = EW(actions,n,theo_LR)
        OPT_pay = OPT(actions)
        ftpl_regret,ew_regret = regret(FTPL_pay,EW_pay,OPT_pay,n)
        if n <20:
            many_actions_fewRounds.append([ftpl_regret,ew_regret])
            LR_fewRoundsManyA.append(theo_LR)
        else:
            many_actions_manyRounds.append([ftpl_regret,ew_regret])
            LR_manyRoundsManyA.append(theo_LR)
        # print("\n")


#few actions few rounds
averages_few_few = np.mean(regrets_fewRounds,axis=0)
#few actions many rounds
averages_few_many = np.mean(regrets_manyRounds,axis=0)
#many actions few rounds
averages_many_few = np.mean(many_actions_fewRounds,axis=0)
#many actions many rounds
averages_many_many = np.mean(many_actions_manyRounds,axis=0)

#few actions few rounds
LR_fewRounds = np.mean(LR_fewRounds)
#few actions many rounds
LR_manyRounds = np.mean(LR_manyRounds)
#many actions few rounds
LR_fewRoundsManyA = np.mean(LR_fewRoundsManyA)
#many actions many rounds
LR_manyRoundsManyA = np.mean(LR_manyRoundsManyA)

print(f'In Datasets with few actions (2-5) and few rounds(0-20), avg FTPL regret = {averages_few_few[0]}, avg EW regret = {averages_few_few[1]}, avg LR = {LR_fewRounds} ')
print(f'In Datasets with few actions (5-50) and many rounds(20-50), avg FTPL regret = {averages_few_many[0]}, avg EW regret = {averages_few_many[1]}, avg LR = {LR_manyRounds}')
print(f'In Datasets with many actions (2-5) and few rounds(2-20), avg FTPL regret = {averages_many_few[0]}, avg EW regret = {averages_many_few[1]}, avg LR = {LR_fewRoundsManyA}')
print(f'In Datasets with many actions (5-50) and many rounds(20-50), avg FTPL regret = {averages_many_many[0]}, avg EW regret = {averages_many_many[1]}, avg LR = {LR_manyRoundsManyA}')
print(f'Empirical optimal LR FTPL is at {analyzeFTPL[j][3]} actions with {analyzeFTPL[j][2]} rounds. Optimal LR = {analyzeFTPL[j][0]/2} Regret = {analyzeFTPL[j][0]}. Theoretical LR = {analyzeFTPL[j][1]}, Expected Regret = {analyzeFTPL[j][4]}')
print(f'Empirical optimal LR EW is at {analyzeEW[k][3]} actions with {analyzeEW[k][2]} rounds. Optimal LR = {analyzeEW[k][0]/2} Regret = {analyzeEW[k][0]}. Theoretical LR = {analyzeEW[k][1]}, Expected Regret = {analyzeEW[k][4]}')
