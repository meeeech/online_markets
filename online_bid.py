import math
import random as rand
import numpy as np

val1 = 97.2
val2 = 31.3

opp_bids = []
def generateData(k_bids,val,n_opp_bids):
    class_bids = [20,55,44,83.6,44,65,1.3,27.4,40,55,1.8,76.8,30,89.3,2.55,33.55,31,1,39,50,23.15,48.1,14.3,49.4,43,63,0.3,25.25,8.3,72,18.1,90.9,35,95,18.3,85.1,4,98,10.35,46.85,0.1,80,20.9,25.9,10,50,44.8,78.7,15.28,51.76,2.55,28.8,20,67,23.35,32.75,10.4,50,0.55,46.7,10.2,50,1,32,14,42.9,20,73,20,50,27.8,50.1,30,45,49.7,50,20,50,20,55,49.3,51,31,73,34,82,42,98,38.4,84.1,8.65,39,10,50,2,50,26.9,98.9,10,64,35,90,34.9999,78.3999,31,77,4.5,58.9,30,75,27,90,26.3,89.7,40,83,10.6,65,23.75,48.35,31,50,17.95,33.5,19,52.3,20,50,19.6,94.1,0,96,30,71,27.3,60,49.5,50.1,14.9,45,10,80,29.3,51,10,91,18.8,86.8,20.5,48,12.69,75.79,40.6,65.6,20,75,10,52,10,45,19,50,29,50,16.7,50,38,70,14,93]
    
    #these are all distinct bids according to val and how many bids are desired
    your_bids = np.linspace(0,val,int(k_bids))

    #setup actions matrix with payoff for each bid in advance. Will be used for OPT and algorithm comparison
    actions=[[0 for a in range(n_opp_bids) ] for b in your_bids]
    for n in range(n_opp_bids):
        #store and remember opponent bid for algorithm use
        opponent_bid = class_bids[rand.randint(0,len(class_bids)-1)]
        opp_bids.append(opponent_bid)
        class_bids.remove(opponent_bid)
        #generate actions payoffs table
        for i,bid in enumerate(your_bids):
            if bid <= opponent_bid:
                actions[i][n] = 0
            else:
                #action matrix to be filled with payoffs instead of 1,0
                actions[i][n] = val-bid
    LR = round(math.sqrt(math.log1p( len(actions) / len(actions[0]) ) ),4) 
    return actions,your_bids, LR



def FTPL (actions,your_bids,LR,val,n_rounds):
    #hallucination is h*(number of times in a row we get tails on an epsilon bias coin ... tails bias = 1-epsilon, h=val)
    tails_bias = (1-LR)

    #FTPL hallucinations generated and allotted according to index, starting sums for each action
    hallucinations=[]
    for action in actions:
        if LR > 1: 
            hallucination = val*np.random.geometric(1-abs(tails_bias))
        else:
            hallucination = val*np.random.geometric(tails_bias)
        hallucinations.append([hallucination])

    actions = np.insert(actions,[0],hallucinations ,axis=1)

    all_picks = []

    for round_idx in range(n_rounds):
        #bid we tested generated action data on before. Now we see if which bid alg will choose based on sums for this opp bid
        random_bid = opp_bids[round_idx]

        #always pick the bid with the highest sum up to this round
        pick_idx = np.argmax(actions,axis=0)[round_idx]

        #value of pick is in the your bids table
        pick = your_bids[pick_idx]

        if pick < random_bid:
            #keep track of which bid was picked, payoff, and pick_index
            all_picks.append([pick,0,pick_idx])
        elif pick > random_bid: 
            all_picks.append([pick,val-pick,pick_idx])
        else:
            coin_flip= random.randint(0,1)
            if coin_flip_a ==1:
                all_picks.append([pick,val-pick,pick_idx])
            else:
                all_picks.append([pick,0,pick_idx])
    #array of every round's bid pick, payoff for that bid, and the index 
    return all_picks

def getMaxIndex(arr):
    max_idx = 0
    for idx,el in enumerate(arr):
        curr_max= arr[max_idx]
        if el>curr_max:
            max_idx= idx
    return max_idx

def OPT(actions, your_bids):
    sums = [0 for i in range(len(actions))]
    for idx,action in enumerate(actions):
        sums[idx] = sum(action)
    #find out which action had the greatest payoff in hindsight
    max_action_index = getMaxIndex(sums)
    #find out which bid this corresponded to
    return your_bids[max_action_index]

def regret(optimal_bid,alg_bid,n_rounds):
    regret_bid = ((optimal_bid-alg_bid)/n_rounds)
    return regret_bid

vals = [val1,val2]
numb_rounds = [2,10,20,50]
for val in vals:
    for k in np.linspace(3,val,5):
        for n in numb_rounds:
            actions, your_bids,LR = generateData(int(k),val,n)
            opt_bid = OPT(actions,your_bids)
            picks = FTPL(actions,your_bids,LR,val,n)
            picks_averages = np.mean(picks,axis=0)
            avg_bid = picks_averages[0]
            avg_payoff = picks_averages[1]
            avg_regret = regret(opt_bid,avg_bid,n)
            print(f'For value = {val}, with {int(k)} action bids and {n} rounds of opposing bids,regret = {avg_regret}')

    