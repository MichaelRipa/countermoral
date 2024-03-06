#! /usr/bin/python3

from random import choice


def generate_triplets(attr_1 : list,attr_2 : list,attr_3 : list, n : int):
    
    triplets = []
    tally = 0
    for i in range(len(attr_1)):
        ai = attr_1[i]
        for j in range(len(attr_2)):
            aj = attr_2[j]
            for k in range(len(attr_3)):
                ak = attr_3[k]
                triplets.append([ai,aj,ak])
                tally += 1
                if tally == n:
                    return triplets
                
    return triplets

def sample_triplets(attr_1 : list,attr_2 : list,attr_3 : list, n : int):

    triplets = []
    for _ in range(n):
        a1 = choice(attr_1)
        a2 = choice(attr_2)
        a3 = choice(attr_3)
        triplets.append([a1,a2,a3])

    return triplets
