#BFS

from collections import deque

def bfs(graph, start, visited):
  queue = deque([start])
  visited[start] = True
  while queue:
    v=queue.popleft()
    print(v,end=' ')
    for i in graph[v]:
      if not visited[i]:
        queue.append(i)
        visited[i] = True

graph = [
  [],
  [2,3,8],
  [1,7],
  [1,4,5],
  [3,5],
  [3,4],
  [7],
  [2,6,8],
  [1,7]
]

visited = [False] * 9

bfs(graph, 1, visited)


import math
from collections import deque

def solution(board):

    # 위, 오른쪽, 아래, 왼쪽 방향 정의
    dy = [0, 1, 0,-1]
    dx = [1, 0, -1,0]
    ylen = len(board)
    xlen = len(board[0])
    queue = deque()
    queue.append((0,0,0,0))
    queue.append((0,0,0,1))

    #갔는지 체크 테이블
    table = [[math.inf for _ in range(xlen)] for _ in range(ylen)]
       
    while queue:
            
        y,x,cost,head = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            ncost = cost + 600 if i != head else cost + 100
            if nx < 0 or nx >= xlen or ny < 0 or ny >= ylen :
                continue
            #벽
            if board[ny][nx] == 1:
                continue
            #뒤로가기 방지
            if abs(i-head) == 2:
                continue
            #값이 같아도 갈수 있게 해야함 >, =
            if board[ny][nx] == 0 and table[ny][nx] >= ncost:
                table[ny][nx] = ncost
                queue.append((ny,nx,ncost,i))
    return table[-1][-1]


print(solution([[0,0,1,0],[0,0,0,0],[0,1,0,1],[1,0,0,0]]))