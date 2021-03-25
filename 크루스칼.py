def find_parent(parent, x):
    if parent[x] != x:
        parent[x] = find_parent(parent,parent[x])
    return parent[x]
def union_parent(parent,a,b):
    a = find_parent(parent,a)
    b = find_parent(parent,b)
    if a < b:
        parent[b] = a
    else: 
        parent[a] = b
def solution(n, costs):

    v = n
    edges = [[cost,a,b] for a,b,cost in costs]
    parent = [0] * (v+1)
    result = 0

    for i in range(1,v+1):
        parent[i] = i
    
    edges.sort()

    for edge in edges:
        cost, a, b = edge
        if find_parent(parent,a) != find_parent(parent,b) :
            union_parent(parent,a,b)
            result += cost
    
    return result