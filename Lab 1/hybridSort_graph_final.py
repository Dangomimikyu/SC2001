#region imports
import random as rand
import matplotlib.pyplot as plt
import time as t;
import math
#endregion

#region sort
class FinalHybridSorter:
    
    def __init__(self, threshold: int):
        self.threshold = threshold  # threshold to switch to insertion sort
        self.key_comparisons = 0   # Counter for key comparisons
        
    
    def reset_keycomparisons(self):
        self.key_comparisons = 0
        
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

    def PureMergeSort(self, arr, left, right):
        # Run mergesort but with threshold = 1 (i.e., never switch to insertion)
        if left >= right:
            return
        mid = (left + right) // 2
        self.PureMergeSort(arr, left, mid)
        self.PureMergeSort(arr, mid + 1, right)
        self.Merge(arr, left, mid, right)

#endregion

#region number generation
def GetArr(size:int, min:int, max:int):
    ret = []
    for i in range(size):
        ret.append(rand.randint(min, max))
    return ret

def fit_c_for_hybrid(n_list, s_value, y_empirical):
    """
    Least-squares fit for c in:
      y ≈ n*log2(n/s) + c*n*s
    """
    numerator = 0.0
    denominator = 0.0
    for n, y in zip(n_list, y_empirical):
        base = n * math.log2(float(n) / float(s_value))
        x = n * s_value
        yprime = y - base
        numerator += x * yprime
        denominator += x * x
    if denominator == 0.0:
        return 0.0
    return numerator / denominator

def fit_c_for_hybrid_vsS(n_value, s_list, y_empirical):
    """
    Least-squares fit for c in:
      y ≈ n*log2(n/s) + c*n*s
    for fixed n and varying s.
    """
    numerator = 0.0
    denominator = 0.0
    for s, y in zip(s_list, y_empirical):
        base = n_value * math.log2(float(n_value) / float(s))
        x = n_value * s
        yprime = y - base
        numerator += x * yprime
        denominator += x * x
    if denominator == 0.0:
        return 0.0
    return numerator / denominator
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

def plot_vs_n_fixedS(S_fixed=8, n_start=1_000, n_stop=100_000, n_step=5_000):
    """
    Figure (i): Key comparisons vs n (S fixed = 8)
    Empirical hybrid + dotted HYBRID theoretical estimate.
    """

    print(f"running {S_fixed}\n")

    sorter = FinalHybridSorter(threshold=S_fixed)
    n_values = list(range(n_start, n_stop + 1, n_step))

    empirical = []
    start = t.perf_counter()
    for n in n_values:
        arr = GetArr(n,0, 10_000_000)
        sorter.reset_keycomparisons()
        sorter.HybridMergeSort(arr, 0, len(arr) - 1)
        empirical.append(sorter.key_comparisons)
    elapsed = t.perf_counter() - start
    print(f"[vs n] S={S_fixed}: ran {len(n_values)} sorts in {elapsed:.2f}s")

    # Theoretical hybrid line (dotted), fit c to empirical
    c_fit = fit_c_for_hybrid(n_values, S_fixed, empirical)
    theory = [int(n * math.log2(float(n) / float(S_fixed)) + c_fit * n * S_fixed) for n in n_values]

    plt.figure()
    plt.plot(n_values, empirical, label=f"Empirical (Hybrid, S={S_fixed})")
    plt.plot(n_values, theory, linestyle="--", label=f"Theoretical Hybrid (c≈{c_fit:.3e})")
    plt.title("Key Comparisons vs n (S fixed)")
    plt.xlabel("n")
    plt.ylabel("Key Comparisons")
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_vs_S_fixedn(n_fixed=10_000_000, S_min=5, S_max=10):
    """
    Figure (ii): Key comparisons vs S (n fixed = 1,000,000)
    Empirical hybrid + dotted HYBRID theoretical estimate.
    """
    s_values = list(range(S_min, S_max + 1))
    empirical = []
    runtimes = []

    start = t.perf_counter()
    for s in s_values:
        sorter = FinalHybridSorter(threshold=s)
        arr = GetArr(n_fixed,0, 10_000_000)
        t0 = t.perf_counter()
        sorter.HybridMergeSort(arr, 0, len(arr) - 1)
        t1 = t.perf_counter()
        empirical.append(sorter.key_comparisons)
        runtimes.append(t1 - t0)
        print(f"[vs S] n={n_fixed}: S={s}, comps={sorter.key_comparisons}, time={t1-t0:.2f}s")
    elapsed = t.perf_counter() - start
    print(f"[vs S] Completed {len(s_values)} runs in {elapsed:.2f}s")

    # Fit c for theoretical hybrid curve for this fixed n
    c_fit = fit_c_for_hybrid_vsS(n_fixed, s_values, empirical)
    theory = [int(n_fixed * math.log2(float(n_fixed) / float(s)) + c_fit * n_fixed * s) for s in s_values]

    plt.figure()
    plt.plot(s_values, empirical, label=f"Empirical (n={n_fixed:,})")
    plt.plot(s_values, theory, linestyle="--", label=f"Theoretical Hybrid (c≈{c_fit:.3e})")
    plt.title("Key Comparisons vs S (n fixed)")
    plt.xlabel("S (threshold)")
    plt.ylabel("Key Comparisons")
    plt.legend()
    plt.grid(True, alpha=0.3)

    return s_values, empirical, runtimes

def plot_optimal_S_vs_n(S_min=1, S_max=20, n_start=5_000, n_stop=100_000, n_step=5_000):
    """
    Figure (iii): Empirical Optimal S* vs n
      - scan S in [S_min..S_max] for each n
      - choose S that minimizes key comparisons
    """
    n_values = list(range(n_start, n_stop + 1, n_step))
    optimal_S = []

    for n in n_values:
        best_s = None
        best_comps = None
        for s in range(S_min, S_max + 1):
            sorter = FinalHybridSorter(threshold=s)
            arr = GetArr(n,0, 10_000_000)
            sorter.reset_keycomparisons()
            sorter.HybridMergeSort(arr, 0, len(arr) - 1)
            comps = sorter.key_comparisons
            if best_comps is None or comps < best_comps:
                best_comps = comps
                best_s = s
        optimal_S.append(best_s)
        print(f"[opt S] n={n} -> S*={best_s}")

    plt.figure()
    plt.plot(n_values, optimal_S, marker='o')
    plt.title("Optimal S vs n (empirical)")
    plt.xlabel("n")
    plt.ylabel("S* (argmin comparisons)")
    plt.grid(True, alpha=0.3)
    return n_values, optimal_S

def plot_partD_barcharts(n_value=100_000, S_fixed=8, S_range_for_opt=(1,20)):
    """
    Part (D): Two bar charts at a fixed n:
      (1) Key comparisons for: pure mergesort, hybrid(S_fixed), hybrid(S*)
      (2) Runtime (seconds) for the same three.
    """
    # Use the same base array across variants for fairness
    base_arr = GetArr(n_value,0, 10_000_000)

    # Pure mergesort
    sorter_pure = FinalHybridSorter(threshold=1)  # threshold value unused here
    sorter_pure.reset_keycomparisons()
    t0 = t.perf_counter()
    sorter_pure.PureMergeSort(base_arr[:], 0, len(base_arr) - 1)
    t1 = t.perf_counter()
    comps_pure = sorter_pure.key_comparisons
    time_pure = t1 - t0

    # Hybrid with fixed S
    sorter_fixed = FinalHybridSorter(threshold=S_fixed)
    sorter_fixed.reset_keycomparisons()
    t0 = t.perf_counter()
    sorter_fixed.HybridMergeSort(base_arr[:], 0, len(base_arr) - 1)
    t1 = t.perf_counter()
    comps_fixed = sorter_fixed.key_comparisons
    time_fixed = t1 - t0

    # Find S* for this n
    s_lo, s_hi = S_range_for_opt
    best_s = None
    best_comps = None
    best_time = None
    for s in range(s_lo, s_hi + 1):
        sorter = FinalHybridSorter(threshold=s)
        sorter.reset_keycomparisons()
        t0 = t.perf_counter()
        sorter.HybridMergeSort(base_arr[:], 0, len(base_arr) - 1)
        t1 = t.perf_counter()
        comps = sorter.key_comparisons
        runtime = t1 - t0
        if best_comps is None or comps < best_comps:
            best_comps = comps
            best_time = runtime
            best_s = s
    print(f"[Part D] For n={n_value}, S*={best_s}, comps={best_comps}, time={best_time:.2f}s")

    # Bar chart 1: key comparisons
    labels = ["Pure MergeSort", f"Hybrid (S={S_fixed})", f"Hybrid (S*={best_s})"]
    comp_values = [comps_pure, comps_fixed, best_comps]

    plt.figure()
    xs = list(range(len(labels)))
    plt.bar(xs, comp_values)
    plt.xticks(xs, labels, rotation=15)
    plt.ylabel("Key Comparisons")
    plt.title(f"Part (D): Key Comparisons (n={n_value:,})")
    plt.grid(axis='y', alpha=0.2)

    # Bar chart 2: runtime
    time_values = [time_pure, time_fixed, best_time]
    plt.figure()
    plt.bar(xs, time_values)
    plt.xticks(xs, labels, rotation=15)
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Part (D): Runtime (n={n_value:,})")
    plt.grid(axis='y', alpha=0.2)
#endregion

#region main
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
    # plot_graph(s_values, comparison_data, step=step)

    # --- Figure (i): vs n (S fixed = 8) with dotted HYBRID theory ---
    plot_vs_n_fixedS(S_fixed=1, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=5, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=10, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=11, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=12, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=20, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=50, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=100, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=150, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)
    plot_vs_n_fixedS(S_fixed=200, n_start=5_000, n_stop=10_000_000, n_step=1_000_000)

    # --- Figure (ii): vs S (n fixed = 1,000,000) with dotted HYBRID theory ---
    # NOTE: This will be very slow in pure Python. Adjust locally if needed.
    # plot_vs_S_fixedn(n_fixed=1_000_000, S_min=1, S_max=20)

    # --- Figure (iii): Optimal S vs n (S in 1..20, n step = 5k) ---
    # plot_optimal_S_vs_n(S_min=8, S_max=13, n_start=5_000, n_stop=100_000, n_step=5_000)

    # --- Part (D): two bar charts (key comparisons + runtime) at a fixed n ---
    # plot_partD_barcharts(n_value=100_000, S_fixed=11, S_range_for_opt=(1,20))

    plt.show()
    print("help")
#endregion

main()