#region imports
import random as rand
#endregion

#region sort
def Merge(l1, l2):
    if l1 == [] or l2 == []:
        return l1 + l2
    
    sortedList = []
    while l1 != [] and l2 != []:
        first1 = l1[0]
        first2 = l2[0]

        if (first1 <= first2):
            sortedList.append(first1)
            l1 = l1[1:]
        else:
            sortedList.append(first2)
            l2 = l2[1:]
    
    return sortedList + l1 + l2

def InsertionSort(arr):
    # just in case
    if (len(arr) <= 1):
        return arr;

    for i in range(1, len(arr)):
        curr = arr[i]

        tempPos = i-1
        while tempPos >= 0 and curr < arr[tempPos]:
            arr[tempPos + 1] = arr[tempPos]
            tempPos -= 1

        arr[tempPos + 1] = curr
    return arr


# let s be the stopping length for smallest arr splits
def HybridMergeSort(arr, s:int):
    if len(arr) <= s:
        # switch to insertion sort
        InsertionSort(arr)
        return arr


    mid = len(arr) // 2
    sub1 = HybridMergeSort(arr[:mid], s)
    sub2 = HybridMergeSort(arr[mid:], s)

    return Merge(sub1, sub2)
#endregion 

#region number generation
def GetArr(size:int, min:int, max:int):
    ret = []
    for i in range(size):
        ret.append(rand.randint(min, max))

    return ret

#endregion

def main():
    minNumber = 1
    maxNumber = 20_000_000 # 20 mil

    # a = []
    # s = 300
    # for i in range(1000, 10_000_000):
    #     a = GetArr(i, minNumber, maxNumber)
    #     HybridMergeSort(a, s)

    a = [1, 5, 2, 6, 7, 4, 8, 3]
    s = 3
    HybridMergeSort(a, s)



main()