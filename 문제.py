from itertools import permutations
def solution(A,B,C,D):
    op = [list(y) for y in permutations([A,B,C,D],4)]     
    op = [str(a) + str(b) +str(':') + str(c) + str(d) for a,b,c,d in op if int(str(a) + str(b)) < 24 and int(str(c) + str(d)) < 60 ]
    print(list(set(op)))
    return len(list(set(op)))

print(solution(0,1,0,1))