import math

def solution(n, s):
    if n > s:
        return [-1]
    
    a = s//n
    b = s%n

    arr = [a] * n

    for i in range(1,b+1):
        arr[-i]+=1
    
    return arr
print(solution(2,9))
