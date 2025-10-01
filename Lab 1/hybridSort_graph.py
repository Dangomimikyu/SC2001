#region imports
import random as rand
import matplotlib.pyplot as plt
#endregion

#region sort
class FinalHybridSorter:
    
    def __init__(self, threshold: int):
        self.threshold = threshold #threshold to find optimal S
        self.key_comparisons = 0  # Counter for key comparisons
        
    def Merge(self, l1, l2):
        self.key_comparisons += 1 #increment key comparison counter
        if l1 == [] or l2 == []:
            return l1 + l2
        
        sortedList = []
        while l1 != [] and l2 != []:
            first1 = l1[0]
            first2 = l2[0]
            self.key_comparisons += 1 #increment key comparison counter
            
            if (first1 <= first2):
                sortedList.append(first1)
                l1 = l1[1:]
            else:
                sortedList.append(first2)
                l2 = l2[1:]
        
        return sortedList + l1 + l2

    def InsertionSort(self,arr):
        self.key_comparisons += 1 #increment key comparison counter
        # just in case
        if (len(arr) <= 1):
            return arr;

        for i in range(1, len(arr)):
            curr = arr[i]

            tempPos = i-1
            while tempPos >= 0 and curr < arr[tempPos]:
                arr[tempPos + 1] = arr[tempPos]
                tempPos -= 1
                self.key_comparisons += 1 #increment key comparison counter

            arr[tempPos + 1] = curr
        return arr

    # let s be the stopping length for smallest arr splits
    def HybridMergeSort(self, arr):
        if len(arr) <= self.threshold:
            # switch to insertion sort
            return self.InsertionSort(arr)

        mid = len(arr) // 2
        sub1 = self.HybridMergeSort(arr[:mid])
        sub2 = self.HybridMergeSort(arr[mid:])

        return self.Merge(sub1, sub2)
#endregion

#region temp
    def insertion_sort(arr, left, right):
        comparisons = 0
        for i in range(left+1, right+1):
            key = arr[i]
            j = i - 1

            while j >= left:
                comparisons += 1
                if arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                else:
                    break
            arr[j + 1] = key
        return comparisons


    def merge_subarray(arr, left, mid, right):
        comparisons = 0
        arr1 = arr[left:mid+1]
        arr2 = arr[mid+1:right+1]
        i = j = 0
        k = left

        while i < len(arr1) and j < len(arr2):
            comparisons += 1
            if arr1[i] < arr2[j]:
                arr[k] = arr1[i]
                i += 1
            else:
                arr[k] = arr2[j]
                j += 1
            k += 1

        while i < len(arr1):
            arr[k] = arr1[i]
            i += 1
            k += 1

        while j < len(arr2):
            arr[k] = arr2[j]
            j += 1
            k += 1
        return comparisons

    def hybrid_sort(arr, left, right, S):
        if left >= right:
            return 0
        if right - left + 1 <= S:
            return insertion_sort(arr, left, right)

        mid = (left + right) // 2
        comparisons = 0
        comparisons += hybrid_sort(arr, left, mid, S)
        comparisons += hybrid_sort(arr, mid+1, right, S)
        comparisons += merge_subarray(arr, left, mid, right)
        return comparisons

    def reset_keycomparisons(self):
        self.key_comparisons = 0
#endregion

#region number generation
def GetArr(size:int, min:int, max:int):
    ret = []
    for i in range(size):
        ret.append(rand.randint(min, max))

    return ret
#endregion

#region plot graph
def plot_graph(s_values, comparison_data, step=5000):
    #Plot the number of key comparisons for different threshold values S
    plt.figure(figsize=(12, 6))
    for i in range(len(s_values)):
        s = s_values[i]
        data = comparison_data[i]
        # Explicitly create array sizes to match data length
        array_sizes = range(1000, len(data)*step + 1000, step)
        plt.plot(array_sizes, data, marker='o', label=f'S={s}')

    plt.xlabel('Array Size (n)')
    plt.ylabel('Key Comparisons')
    plt.title('Hybrid Mergesort Key Comparisons vs Array Size (n)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    #adjust the s values in different comparisons to test it out
    s_values = [11]  # Threshold values
    comparison_data = []

    #parameters for testing
    minNumber = 1
    maxNumber = 20_000_000 # 20 mil

    #adjust the maximum array size
    maxSize = 10_000_000 #10 mil elements
    step = 1_000_000 #step size for increasing increments

    for s in s_values:
        sorter = FinalHybridSorter(threshold=s)
        comparisons_for_s = []

        for size in range(1_000_000, maxSize + 1, step):
            print(f"Sorting array of size {size} with S={s}...")    
            arr = GetArr(size, minNumber, maxNumber)
            sorter.reset_keycomparisons()
            sorter.HybridMergeSort(arr)
            comparisons_for_s.append(sorter.key_comparisons)

        comparison_data.append(comparisons_for_s)

    plot_graph(s_values, comparison_data, step=step)
    print(f"Program completed")



main()
