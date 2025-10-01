#region imports
import random as rand
import matplotlib.pyplot as plt
import time as t;
#endregion

#region sort
class FinalHybridSorter:
    
    def __init__(self, threshold: int):
        self.threshold = threshold  # threshold to switch to insertion sort
        self.key_comparisons = 0   # Counter for key comparisons
        
    def InsertionSort(self, arr, left, right):
        for i in range(left + 1, right + 1):
            curr = arr[i]
            j = i - 1
            
            while j >= left:
                self.key_comparisons += 1
                if arr[j] > curr:
                    arr[j + 1] = arr[j]
                    j -= 1
                else:
                    break
            arr[j + 1] = curr

    def Merge(self, arr, left, mid, right):
        left_sub = arr[left:mid + 1]
        right_sub = arr[mid + 1:right + 1]
        
        i = 0
        j = 0
        k = left
        
        while i < len(left_sub) and j < len(right_sub):
            self.key_comparisons += 1
            if left_sub[i] <= right_sub[j]:
                arr[k] = left_sub[i]
                i += 1
            else:
                arr[k] = right_sub[j]
                j += 1
            k += 1
        
        # Copy any remaining elements
        while i < len(left_sub):
            arr[k] = left_sub[i]
            i += 1
            k += 1
            
        while j < len(right_sub):
            arr[k] = right_sub[j]
            j += 1
            k += 1

    def HybridMergeSort(self, arr, left, right):
        if left >= right:
            return
        
        if right - left + 1 <= self.threshold:
            self.InsertionSort(arr, left, right)
            return
        
        mid = (left + right) // 2
        self.HybridMergeSort(arr, left, mid)
        self.HybridMergeSort(arr, mid + 1, right)
        self.Merge(arr, left, mid, right)

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
    startTime = t.perf_counter()

    #adjust the s values in different comparisons to test it out
    s_values = [11]  # Threshold values
    comparison_data = []

    #parameters for testing
    minNumber = 1
    maxNumber = 20_000_000 # 20 mil

    #adjust the maximum array size
    maxSize = 100000 #10 mil elements
    step = 10000 #step size for increasing increments

    for s in s_values:
        sorter = FinalHybridSorter(threshold=s)
        comparisons_for_s = []

        for size in range(1000, maxSize + 1, step):
            print(f"Sorting array of size {size} with S={s}...")    
            arr = GetArr(size, minNumber, maxNumber)
            sorter.reset_keycomparisons()
            sorter.HybridMergeSort(arr, 0, len(arr)-1)
            comparisons_for_s.append(sorter.key_comparisons)

        comparison_data.append(comparisons_for_s)

    endTime = t.perf_counter()
    dur = endTime - startTime
    print(f"Program completed with time: {dur:.2f}s")
    plot_graph(s_values, comparison_data, step=step)



main()
