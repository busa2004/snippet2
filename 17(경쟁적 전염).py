from collections import deque
n,k = 3,3
data = [[1,0,2],[0,0,0],[3,0,0]]
gs,gx,gy = 1,2,2
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]
v_arr = []
second = 0
for i in range(n):
  for j in range(k):
    if data[i][j] != 0:
      v_arr.append([data[i][j],i,j])

v_arr = sorted(v_arr, key = lambda x : x[0])

queue = deque(v_arr)

while(second < gs):
  second += 1
  length = len(queue)
  for _ in range(length):
    v,i,j  = queue.popleft()
    for d in range(4):
      x = i + dx[d]
      y = j + dy[d]
      if x >= 0 and x < n and y >= 0 and y < k and data[x][y] == 0:
        data[x][y] = v
        queue.append([v,x,y])      

print(data[gx-1][gy-1])
