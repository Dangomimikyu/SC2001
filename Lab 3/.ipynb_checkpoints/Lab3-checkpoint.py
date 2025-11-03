import random
import time
import math

def MakeObjList(size: int):
    objs = [[] * size]
    for i in range(size):
        # weight
        objs[i][0] = random.randint(0, size - 1)
        # profit
        objs[i][1] = random.randint(0, size - 1)

# return: max profit
def P(capacity:int) -> int:
    pass

if __name__ == "__main__":
    knapsack = MakeObjList(3)

    # find max profit for bag of size C = 15    
    P(15)