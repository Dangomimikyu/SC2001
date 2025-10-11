import random
import time
import math
import heapq
import matplotlib.pyplot as plt

#region graph building
# =========================
# Graph builders
# =========================

def make_adjacency_matrix(n, m):
    # Check for self loops and no duplicate edges
    if m > (n * (n - 1)) // 2:
        raise ValueError("Too many edges for a simple undirected graph.")

    # initialize matrix of zeros (n x n)
    M = [[0] * n for _ in range(n)]

    # Track used undirected edges as a list of (a,b) with a<b
    used_edges = []

    while len(used_edges) < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue

        if u < v:
            a = u
            b = v
        else:
            a = v
            b = u

        # check duplicate
        found = False
        k = 0
        while k < len(used_edges):
            if used_edges[k][0] == a and used_edges[k][1] == b:
                found = True
                break
            k += 1
        if found:
            continue

        w = random.randint(1, 20)
        used_edges.append((a, b))
        M[a][b] = w
        M[b][a] = w

    return M

def matrix_to_adjlist(M):
    #Convert adjacency matrix → adjacency list for the EXACT same graph (undirected).
    n = len(M)

    adj = []
    i = 0
    while i < n:
        adj.append([])
        i += 1

    u = 0
    while u < n:
        v = u + 1
        while v < n:
            w = M[u][v]
            if w != 0:
                adj[u].append((v, w))
                adj[v].append((u, w))
            v += 1
        u += 1

    return adj
#endregion

#region dijkstra implementation
# =========================
# Dijkstra implementations
# =========================

def dijkstra_matrix_array(M, start=0):
    #(a) Adjacency Matrix + array scan Priority queue.
    n = len(M)

    dist = []
    vis = []
    i = 0
    while i < n:
        dist.append(float('inf'))
        vis.append(False)
        i += 1
    dist[start] = 0

    iters = 0
    while iters < n:
        # extract-min
        u = -1
        best = float('inf')
        i = 0
        while i < n:
            if (not vis[i]) and dist[i] < best:
                best = dist[i]
                u = i
            i += 1
        if u == -1:
            break
        vis[u] = True

        # update neighbors if a shorter path is found
        v = 0
        while v < n:
            w = M[u][v]
            if w != 0 and (not vis[v]):
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
            v += 1
        iters += 1

    return dist


def dijkstra_adjlist_heap(adj, start=0):
    #(b) Adjacency list + min-heap Priority Queue (heapq)
    n = len(adj)

    dist = []
    vis = []
    i = 0
    while i < n:
        dist.append(float('inf'))
        vis.append(False)
        i += 1
    dist[start] = 0

    pq = []
    heapq.heappush(pq, (0, start))

    while len(pq) > 0:
        item = heapq.heappop(pq)
        d_u = item[0]
        u = item[1]

        if vis[u]:
            continue
        vis[u] = True

        # iterate neighbors
        j = 0
        while j < len(adj[u]):
            v = adj[u][j][0]
            w = adj[u][j][1]
            if not vis[v]:
                nd = d_u + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
            j += 1

    return dist
#endregion

#region util functions
# =========================
# Timing & theory helpers
# =========================

def m_of_n(n):
    # Choose |E| as a function of |V| to ensure sparse but connected graphs.
    # ensure at least n-1 edges for a tendency toward connectivity
    # larger c denotes fewer edges → sparser
    # lower c denotes more edges → denser
    c = 4
    m_candidate = c * n
    if m_candidate < n - 1:
        return n - 1
    return m_candidate


def measure_both_on_same_graphs(n_list, trials):
    
    # trials is used for averaging time over multiple runs
    '''
    For each n and each trial:
      - Build ONE matrix graph with m_of_n(n) edges
      - Convert to list (same edges)
      - Time (a) on matrix, (b) on list
    '''

    A_pts = []
    B_pts = []

    idx = 0
    while idx < len(n_list):
        n = n_list[idx]
        m = m_of_n(n)
        e_dir = m
        sumA = 0.0
        sumB = 0.0

        t = 0
        while t < trials:
            M = make_adjacency_matrix(n, m)
            adj = matrix_to_adjlist(M)

            t0 = time.perf_counter() 
            _ = dijkstra_matrix_array(M, start=0) # _ is used here to ignore output
            sumA += time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = dijkstra_adjlist_heap(adj, start=0) # _ is used here to ignore output
            sumB += time.perf_counter() - t0

            t += 1

        A_pts.append((n, sumA / trials, e_dir))
        B_pts.append((n, sumB / trials, e_dir))
        print("n={:4d} |E|≈{:6d}  [A] {:.5f}s  [B] {:.5f}s".format(
            n, e_dir, A_pts[-1][1], B_pts[-1][1]
        ))

        idx += 1

    return A_pts, B_pts


def scaled_theory_A(points):
    # O(V^2), scaled to the first empirical point
    vs = []
    ys = []
    i = 0
    while i < len(points):
        vs.append(points[i][0])
        ys.append(points[i][1])
        i += 1

    if len(vs) == 0:
        return [], []

    v0 = vs[0]
    base0 = v0 * v0
    if base0 > 0:
        c = ys[0] / base0
    else:
        c = 1.0

    y_theory = []
    i = 0
    while i < len(vs):
        v = vs[i]
        y_theory.append(c * (v * v))
        i += 1

    return vs, y_theory


def scaled_theory_B(points):
    #O((V+E) log V), scaled to the first empirical point
    vs = []
    Es = []
    ys = []
    i = 0
    while i < len(points):
        vs.append(points[i][0])
        ys.append(points[i][1])
        Es.append(points[i][2])
        i += 1

    if len(vs) == 0:
        return [], []

    v0 = vs[0]
    e0 = Es[0]
    base_log = math.log(max(2, v0), 2) #log2 V
    base0 = (v0 + e0) * base_log #(V+E) * log2(V) first point
    if base0 > 0:
        c = ys[0] / base0
    else:
        c = 1.0

    y_theory = []
    i = 0
    while i < len(vs):
        v = vs[i]
        e = Es[i]
        val = (v + e) * math.log(max(2, v), 2) # (V+E) log(V)
        y_theory.append(c * val) # c * (V+E) log(V) to scale the points
        i += 1

    return vs, y_theory

def get_vertex_list(start: int, stop: int, step:int) -> list:
    ret = []
    for i in range(start, stop + 1, step):
        ret.append(i)
    
    return ret
#endregion

#region part a
# =========================
# Part (a) plot
# =========================

def run_part_a_plot(A_pts):
    vs = [] #x-axis
    emp = [] #y-axis
    i = 0
    while i < len(A_pts):
        vs.append(A_pts[i][0])
        emp.append(A_pts[i][1])
        i += 1

    vs_th, th = scaled_theory_A(A_pts)

    plt.figure(figsize=(9, 5))
    plt.plot(vs, emp, 'o-', label="Empirical (matrix + array)")
    plt.plot(vs_th, th, '--', label="Scaled theory ~ O(V^2)")
    plt.xlabel("|V|")
    plt.ylabel("Time (seconds)")
    plt.title("Part (a): Dijkstra (Adjacency Matrix + Array)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
#endregion

#region part b
# =========================
# Part (b) plot
# =========================
def run_part_b_plot(B_pts):
    vs = [] #x-axis
    emp = [] #y-axis
    i = 0
    while i < len(B_pts):
        vs.append(B_pts[i][0])
        emp.append(B_pts[i][1])
        i += 1

    vs_th, th = scaled_theory_B(B_pts)

    plt.figure(figsize=(9, 5))
    plt.plot(vs, emp, 'o-', label="Empirical (list + heap)")
    plt.plot(vs_th, th, '--', label="Scaled theory ~ O((V+E)log V)")
    plt.xlabel("|V|")
    plt.ylabel("Time (seconds)")
    plt.title("Part (b): Dijkstra (Adjacency List + Min-Heap)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
#endregion

# =========================
# Main
# =========================

if __name__ == "__main__":
    # Test sizes and trials
    # n_list = [200, 400, 800, 1600, 3200]
    # n_list = get_vertex_list(10, 100, 10);
    n_list = get_vertex_list(1000, 10000, 1000)

    trials = 3

    # Measure both on the same graphs (fair)
    A_pts, B_pts = measure_both_on_same_graphs(n_list, trials)

    # Part (a): empirical vs theory
    run_part_a_plot(A_pts)

    # Part (b): empirical vs theory
    run_part_b_plot(B_pts)
