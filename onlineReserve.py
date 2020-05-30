import math
import random as rand
import numpy as np

bids = np.random.randint(0,1)

def genData(n_bidders,opt_Res):
    all_bidders = []
    for n in range(n_bidders):
        #making the number of rounds consistent at 100
        bidder_n_bids= [rand.uniform(0,1) for i in range(100)]
        all_bidders.append(bidder_n_bids)

    reserve_prices = [round(rand.uniform(0,1),3) for i in range(20)]
    actions_matrix= [[0 for j in range(100)]for k in range(20)]
    #ensure the optimal reserve is a possible action price for the algorithm to choose
    if opt_Res not in reserve_prices:
        reserve_prices.remove(reserve_prices[rand.randint(0,len(reserve_prices)-1)])
        reserve_prices.insert(rand.randint(0,len(reserve_prices)-1),opt_Res)
    print("RESERVE PRICE ACTION SPACE", reserve_prices)
    for idx,bid in enumerate(all_bidders[0]):
        for i,action in enumerate(reserve_prices):
            if all_bidders[0][idx] == reserve_prices[i] or all_bidders[1][idx] == reserve_prices[i]:
                rev = reserve_prices[i]
                # rev = min(all_bidders[0][idx] , all_bidders[1][idx])
            if all_bidders[0][idx] > reserve_prices[i] and all_bidders[1][idx] > reserve_prices[i]:
                rev = min(all_bidders[0][idx] , all_bidders[1][idx])
            elif all_bidders[0][idx] < reserve_prices[i] and all_bidders[1][idx] < reserve_prices[i]:
                rev = 0
            else:
                rev = reserve_prices[i]
                # rev = ((1- reserve_prices[i])/3) + reserve_prices[i]
            # print("len", len(actions_matrix),len(actions_matrix[0]))
            # print("i, idx", i, idx)
            # print('res at 20', reserve_prices[20])
            rev = round(rev,3)
            actions_matrix[i][idx] = round(rev,3)
    # print("PRICES", reserve_prices)
    # print("ACTIONS", actions_matrix)
    LR = round(math.sqrt(math.log1p( len(actions_matrix) / len(actions_matrix[0]) ) ),4) 
    return all_bidders, reserve_prices, actions_matrix,LR

def calculateOpt(n_bidders):
    optReservePrice = 1/2
    ExpRev = 5/12
    return optReservePrice, ExpRev

def genData_multipleBidders(n_bidders,opt_Res):
    all_bidders = []
    for n in range(n_bidders):
        #making the number of rounds consistent at 100
        bidder_n_bids= [rand.uniform(0,1) for i in range(100)]
        all_bidders.append(bidder_n_bids)

    reserve_prices = [round(rand.uniform(0,1),3) for i in range(20)]
    actions_matrix= [[0 for j in range(100)]for k in range(20)]
    #ensure the optimal reserve is a possible action price for the algorithm to choose
    if opt_Res not in reserve_prices:
        reserve_prices.remove(reserve_prices[rand.randint(0,len(reserve_prices)-1)])
        reserve_prices.insert(rand.randint(0,len(reserve_prices)-1),opt_Res)
    print("RESERVE PRICE ACTION SPACE", reserve_prices)
    for idx,bid in enumerate(all_bidders[0]):
        all_bidders = np.array(all_bidders)
        # print("ALL BIDS", all_bidders)
        this_round_bids = all_bidders[:,idx]
        # if idx ==0:
            # print("ROUND 0 BIDS", this_round_bids)
        top_2 = highest_bids(this_round_bids)
        # if idx ==0:
        #     print("HIGHEST 2 BIDS round 0: ", top_2)
        for i,action in enumerate(reserve_prices):
            if top_2[0] == reserve_prices[i] or top_2[1] == reserve_prices[i]:
                rev = reserve_prices[i]
                # rev = min(all_bidders[0][idx] , all_bidders[1][idx])
            if top_2[0] > reserve_prices[i] and top_2[1] > reserve_prices[i]:
                rev = min(top_2[0] , top_2[1])
            elif top_2[0] < reserve_prices[i] and top_2[1] < reserve_prices[i]:
                rev = 0
            else:
                rev = reserve_prices[i]
                # rev = ((1- reserve_prices[i])/3) + reserve_prices[i]
            # print("len", len(actions_matrix),len(actions_matrix[0]))
            # print("i, idx", i, idx)
            # print('res at 20', reserve_prices[20])
            rev = round(rev,3)
            actions_matrix[i][idx] = round(rev,3)
    # print("PRICES", reserve_prices)
    # print("ACTIONS", actions_matrix)
    LR = round(math.sqrt(math.log1p( len(actions_matrix) / len(actions_matrix[0]) ) ),4) 
    return all_bidders, reserve_prices, actions_matrix,LR

def highest_bids(arr):
    highest = np.max(arr)
    new_arr = np.delete(arr,np.where(arr == highest))
    second_highest = np.max(new_arr)
    return [highest,second_highest]
# def generateData(k_bids,val,n_opp_bids):    
#     #these are all distinct bids according to val and how many bids are desired
#     your_bids = np.linspace(0,val,int(k_bids))

#     #setup actions matrix with payoff for each bid in advance. Will be used for OPT and algorithm comparison
#     actions=[[0 for a in range(n_opp_bids) ] for b in your_bids]
#     for n in range(n_opp_bids):
#         #store and remember opponent bid for algorithm use
#         opponent_bid = class_bids[rand.randint(0,len(class_bids)-1)]
#         opp_bids.append(opponent_bid)
#         class_bids.remove(opponent_bid)
#         #generate actions payoffs table
#         for i,bid in enumerate(your_bids):
#             if bid <= opponent_bid:
#                 actions[i][n] = 0
#             else:
#                 #action matrix to be filled with payoffs instead of 1,0
#                 actions[i][n] = val-bid
#     LR = round(math.sqrt(math.log1p( len(actions) / len(actions[0]) ) ),4) 
#     return actions,your_bids, LR



def FTPL (actions,reserve_prices,opp_bids,LR,n_rounds,opt_Reserve):
    #hallucination is h*(number of times in a row we get tails on an epsilon bias coin ... tails bias = 1-epsilon, h=val)
    tails_bias = (1-LR)

    #FTPL hallucinations generated and allotted according to index, starting sums for each action
    hallucinations=[]
    for action in actions:
        if LR > 1: 
            hallucination = np.random.geometric(1-abs(tails_bias))
        else:
            hallucination = np.random.geometric(tails_bias)
        hallucinations.append([hallucination])

    actions = np.insert(actions,[0],hallucinations ,axis=1)

    all_revenues = []
    opp_bids = np.array(opp_bids)
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
        #always pick the bid with the highest sum up to this round
        # pick_idx = np.argmax(actions,axis=0)[round_idx]

        #value of pick is in the your bids table
        price = reserve_prices[pick_idx]
        if price == opt_Reserve:
            print(f"FTPL, optimal reserve price reached round: {round_idx}")
        if top_2[0] == price or top_2[1] == price:
            # if bid_1 > price or bid_2 > price:
            #     rev = price
            # else:
            #     rev = bid_1 if bid_1 < price else bid_2
            # rev = min(bid_1 , bid_2)
            rev = price
            # print("price,b1,b2", price,top_2[0],top_2[1],rev)
        elif top_2[0] > price and top_2[1] > price:
            # rev = ((1- price)/3) + price
            rev = min(top_2[0] , top_2[1])
            # print("price,b1,b2", price,top_2[0],top_2[1],rev)
        elif top_2[0] < price and top_2[1] < price:
            # rev = max(bid_1 , bid_2)
            rev = 0
            # print("price,b1,b2", price,top_2[0],top_2[1],rev)
        else:
            # print("PRICE, B1, B2", price,bid_1,bid_2,rev)
            # rev = ((1- price)/3) + price
            # rev = min(bid_1 , bid_2)
            rev = price
            # print("price,b1,b2", price,top_2[0],top_2[1],rev)
        all_revenues.append([rev,price])
        actions[pick_idx][round_idx+1] += rev
    return all_revenues

# def OPT(actions, your_bids,reserve_prices):
#     sums = [0 for i in range(len(actions))]
#     for idx,action in enumerate(actions):
#         sums[idx] = sum(action)
#     #find out which action had the greatest payoff in hindsight
#     opt_action_idx = np.argmax(sums)
#     #find out which bid this corresponded to
#     return reserve_prices[opt_action_idx]

def EW(actions,n_rounds, LR,reserve_prices,opp_bids,opt_Reserve):
    all_revenues = []
    probabilities=[0 for i in range(len(actions))]
    idx = rand.randint(0,len(actions)-1)
    opp_bids = np.array(opp_bids)
    for round_idx in range(n_rounds):
        if round_idx == 0:
            price = reserve_prices[idx]
        else:
            for i,action in enumerate(actions):
                top = math.pow(1+LR, sum(action[:round_idx]))
                bottom = getSumAllOtherActionProbs(actions,round_idx,LR)
                action_prob = round(top/bottom,4)
                probabilities[i] = action_prob
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()
            pick_idx = np.random.choice([i for i in range(len(actions))],p=probabilities)
            # pick = actions[pick_idx][round_idx]
            price = reserve_prices[pick_idx]
        this_round_bids = opp_bids[:,round_idx]
        top_2 = highest_bids(this_round_bids)
        # bid_1 = opp_bids[0][round_idx]
        # bid_2 = opp_bids[1][round_idx]
        if price == opt_Reserve:
            print(f"EW, optimal reserve price reached round: {round_idx}")
        if top_2[0] == price or top_2[1] == price:
            rev = price
        elif top_2[0] > price and top_2[1] > price:
            rev = min(top_2[0] , top_2[1])
        elif top_2[0] < price and top_2[1] < price:
            rev = 0
        else:
            rev = price
        all_revenues.append([rev,price])
        # payoff+=pick
    # print(f"EW payoff = {payoff}")
    return all_revenues

def getSumAllOtherActionProbs(arr,round_idx,LR):
    action_probs = []
    for i,each_arr in enumerate(arr):
        curr_action_prob = math.pow(1+LR, sum(each_arr[:round_idx]))
        action_probs.append(curr_action_prob)
    sum_action_probs = sum(action_probs)
    return sum_action_probs

m_bidders = [2,5,10,20]

for m in m_bidders:
    print(f"\n\n\nDISPLAYING RESULTS FOR {m} BIDDERS")
    optRes, ExpRev = calculateOpt(2)
    bids,reserve_prices,actions,LR = genData_multipleBidders(m,optRes)
    revenues_FTPL = FTPL(actions,reserve_prices,bids,LR,100,optRes)
    revenues_FTPL = np.array(revenues_FTPL)
    pays_FTPL = revenues_FTPL[:,0]
    reserves_FTPL = revenues_FTPL[:,1]
    # for i,pay in enumerate(pays_FTPL):
    #     print(f"Round {i+1}, Reserve Price: {reserves_FTPL[i]} , Revenue: {pay}")
    print(f"FTPL AVG Revenue: {round(np.mean(pays_FTPL),3)}") #Expected Revenue: {round(ExpRev,3)}")
    print(f"FTPL AVG Reserve Price: {round(np.mean(reserves_FTPL),3)}, Expected Optimal Reserve Price: {optRes}\n")
    revenues_EW = EW(actions,100,LR,reserve_prices,bids,optRes)
    revenues_EW = np.array(revenues_EW)
    pays_EW = revenues_EW[:,0]
    reserves_EW = revenues_EW[:,1]
    # for i,pay in enumerate(pays_EW):
    #     print(f"Round {i+1}, Reserve Price: {reserves_EW[i]} , Revenue: {pay}")
    print(f"EW AVG Revenue: {round(np.mean(pays_EW),3)}")# Expected Revenue: {round(ExpRev,3)}")
    print(f"EW AVG Reserve Price: {round(np.mean(reserves_EW),3)}, Expected Optimal Reserve Price: {optRes}")
# print("RESERVE PRICES CHOSEN", revenues[:,1])
