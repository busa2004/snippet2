#택시 합승 문제 
def solution(n, s, a, b, fares):
    arr= [[20000001] * n for _ in range(n)]

    for i,j,v in fares:
        arr[i-1][j-1] = v
        arr[j-1][i-1] = v

    for i in range(n) :
        arr[i][i] = 0

    for i in range(n):
        for j in range(n):
            for k in range(n): 
                if arr[j][k] > arr[j][i] + arr[i][k]:
                    arr[j][k] = arr[j][i] + arr[i][k]
    minv = 40000001
    for i in range(len(arr)):
        minv = min(minv , arr[s-1][i] + arr[i][a-1] + arr[i][b-1])
    return minv