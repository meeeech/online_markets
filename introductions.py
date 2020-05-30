
#create learning algorithm w action space 'introduce' or 'not introduce'
    # action matrix filled w payoff, not introduce all 0, introduce is virtual welfare

import math
import random as rand
import numpy as np


def genValues(n_rounds):
    bidder_values=[]
    for i in range(2):
        bidder= [rand.uniform(0,1) for n in range(n_rounds)]
        bidder_values.append(bidder)
    return bidder_values


def calculateVirtualWelfare(v1,v2):
    return round( ((2*v1) -1)+ ((2*v2) -1) , 3)


def genActions(bidder_values):
    bidder_values = np.array(bidder_values)
    action_matrix = [[0 for i in range(len(bidder_values[0]))] for j in range(2)]
    for k in range(len(bidder_values[0])):
        this_round_bids =  bidder_values[:,k]
        vw = calculateVirtualWelfare(this_round_bids[0],this_round_bids[1])
        action_matrix[0][k] = 0 if vw < 0 else vw
        action_matrix[1][k] = abs(vw) if vw <0 else 0
        # action_matrix[0][k] = calculateVirtualWelfare(this_round_bids[0],this_round_bids[1])
    # print("ACITON MATRIX", action_matrix)
    return action_matrix
def highest_bids(arr):
    highest = np.max(arr)
    new_arr = np.delete(arr,np.where(arr == highest))
    second_highest = np.max(new_arr)
    return [highest,second_highest]

    
def FTPL (actions,opp_bids,LR,n_rounds):
    #hallucination is h*(number of times in a row we get tails on an epsilon bias coin ... tails bias = 1-epsilon, h=val)
    tails_bias = (1-LR)

    #FTPL hallucinations generated and allotted according to index, starting sums for each action
    hallucinations=[]
    for action in actions:
        if LR > 1: 
            hallucination = np.random.geometric(1-abs(tails_bias))
        else:
            hallucination = np.random.geometric(tails_bias)
        # print("HALL", hallucination)
        hallucinations.append([hallucination/5])

    actions = np.insert(actions,[0],hallucinations ,axis=1)

    all_revenues = []
    opp_bids = np.array(opp_bids)
    actions = np.array(actions)
    unsummed_action_payoffs = actions.copy()
    for round_idx in range(n_rounds):
        #bid we tested generated action data on before. Now we see if which bid alg will choose based on sums for this opp bid
        this_round_bids = opp_bids[:,round_idx]
        top_2 = highest_bids(this_round_bids)
        # bid_1 = opp_bids[0][round_idx]
        # bid_2 = opp_bids[1][round_idx]
        all_sums = []
        for ac in actions:
            # print("AC", ac)
            ac = np.array(ac)
            until_this_round = ac[:round_idx+1]
            # print("all until round",round_idx, until_this_round)
            sum_this = sum(until_this_round)
            # print("sum", sum_this)
            all_sums.append(sum_this)
        
        # print("all sums", all_sums)
        #always pick the bid with the highest sum up to this round
        # pick_idx = np.argmax(actions,axis=0)[round_idx]
        pick_idx = np.argmax(all_sums)
        # print("pick", all_sums, pick_idx)
        #value of pick is in the your bids table
        # price = reserve_prices[pick_idx]
        pick = 'introduce' if pick_idx == 0 else "don't introduce"
        # if price == opt_Reserve:
        #     print(f"FTPL, optimal reserve price reached round: {round_idx}")
        rev = actions[pick_idx][round_idx]
        welf = unsummed_action_payoffs[pick_idx][round_idx]
        # print("REV", rev, actions[pick_idx])
        actions[pick_idx][round_idx+1] += rev
        alternate = unsummed_action_payoffs[abs(1-pick_idx)][round_idx]
        all_revenues.append([welf,pick,alternate])
    return all_revenues, unsummed_action_payoffs

bidder_vals = genValues(10)
actions = genActions(bidder_vals)
LR = round(math.sqrt(math.log1p( len(actions) / len(actions[0]) ) ),3) 
revs, actions_w_halluc= FTPL(actions,bidder_vals,LR,10)
# for i,rev in enumerate(revs):
#     print("Revenue: {rev[0]}, Outcome: {rev[1]}, Alternative: {actions[]}", )
# print("REVS",revs)
# print("ACTIONS", actions_w_halluc)
# count = 0
# count_b = 0
# for a in actions_w_halluc[0]:
#     if a !=0:
#         count+=1
results = []
for b in revs:
    results.append(b[1])
expected = []
for a in actions_w_halluc[0]:
    if a >0:
        expected.append('introduce')
    else:
        expected.append("don't introduce")
revs = np.array(revs)

id1 = sum(actions_w_halluc[0])
id2 = sum(actions_w_halluc[1])
opt_rev = id1 if id1 > id2 else id2


opt = 0
actions_w_halluc = np.array(actions_w_halluc)
# print("ALL AC", actions_w_halluc)
intro = actions_w_halluc[0,:]
n_intro = actions_w_halluc[1,:]
# print("INTRO", intro)
# print("NOT INTRO", n_intro)
for i,val in enumerate(intro):
    if intro[i] > n_intro[i]:
        opt += intro[i]
    else:
        opt+= n_intro[i]

print("RESULTS", results)
print("EXPECTED", expected)
print("\nRevenue gained, introduction, and revenue missed:")
print(revs)
print(f"OPT Rev: {opt}, Best in Hindsight Rev:{opt_rev}, FTPL Rev: {sum(revs[:,0].astype(float))} ")
# print(f"Algorithm percentage Introductions {count_b/len(revs)}, Correct Percentage of Introductions{round(count/len(actions_w_halluc[0]),2)} ")
