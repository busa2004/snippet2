#줄서는방법
#순열의 순서를 정확히 찾아낼 수 있음
import math

def solution(n, k):
    answer = []
    numberList = [i for i in range(1, n+1)]
    while (n != 0):
        temp = math.factorial(n) // n # 한개에 몇개씩의 값이 있을지 알 수 잇음.
        index = k // temp
        k = k % temp
        if k == 0:
            answer.append(numberList.pop(index-1))
        else :
             answer.append(numberList.pop(index))

        n -= 1
    
    return answer
print(solution(3,5))


# 회전
    if cur1[Y] == cur2[Y]: # 가로방향 일 때
        UP, DOWN = -1, 1
        for d in [UP, DOWN]:
            if new_board[cur1[Y]+d][cur1[X]] == 0 and new_board[cur2[Y]+d][cur2[X]] == 0:
                cand.append((cur1, (cur1[Y]+d, cur1[X])))
                cand.append((cur2, (cur2[Y]+d, cur2[X])))
    else: # 세로 방향 일 때
        LEFT, RIGHT = -1, 1
        for d in [LEFT, RIGHT]:
            if new_board[cur1[Y]][cur1[X]+d] == 0 and new_board[cur2[Y]][cur2[X]+d] == 0:
                cand.append(((cur1[Y], cur1[X]+d), cur1))
                cand.append(((cur2[Y], cur2[X]+d), cur2))
weak_point = weak + [w+n for w in weak]  # 선형으로
#다익스트라
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



from collections import defaultdict
from bisect import insort, bisect_left
s = 'abc'
s.find(찾을문자, 찾기시작할위치) return 인덱스
s.startswith(시작하는문자, 시작지점) return False
s.endswith(끝나는문자, 문자열의시작, 문자열의끝)

#정규식
match()
search()
findall()
finditer()
import re
p = re.compile('[a-z]+')
m = p.math("python")
print




#자리수 정렬
def solution(numbers):
    numbers = list(map(str, numbers))
    numbers.sort(key=lambda x: x*3, reverse=True)
    return str(int(''.join(numbers)))


# 파이썬
import heapq
from collections import deque
def solution(jobs):
    N, REQUEST = len(jobs), 0
    
    jobs = deque(sorted(jobs))
    jobs_done, curr_time, waits, cand = 0, 0, 0, []
    # 일을 다 마칠 때 까지
    while jobs_done < N:
        # 요청이 들어온 것이 없을 때
        if not cand:
            request, time = jobs.popleft()
            curr_time = request + time
            waits += time
        # 요청이 들어온 것이 있을 때
        else:
            time, request = heapq.heappop(cand)
            curr_time += time
            waits += curr_time - request

        jobs_done += 1
            
        while jobs and jobs[0][REQUEST] <= curr_time:
            heapq.heappush(cand, jobs.popleft()[::-1])
  
    return waits // N
        
        
    


solution([ [2, 6] ,[0, 3], [1, 9]])


import heapq
heap = []
heapq.heappush(heap,5)
heapq.heappop(heap)
print(heap)
#힙 정렬
import heapq

def heap_sort(nums):
  heap = []
  for num in nums:
    heapq.heappush(heap, num)
  
  sorted_nums = []
  while heap:
    sorted_nums.append(heapq.heappop(heap))
  return sorted_nums

print(heap_sort([4, 1, 7, 3, 8, 5]))

#n 번째 힙
import heapq

def kth_smallest(nums, k):
  heap = []
  for num in nums:
    heapq.heappush(heap, num)

  kth_min = None
  for _ in range(k):
    kth_min = heapq.heappop(heap)
  return kth_min

print(kth_smallest([4, 1, 7, 3, 8, 5], 3))

#최대 힙
for num in nums:
  heapq.heappush(heap, (-num, num))  # (우선 순위, 값)

while heap:
  print(heapq.heappop(heap)[1])  # index 1

#heapq 우선순위 큐처럼 전체 정렬이 아니기 때문에 더 빠름
import heapq
heap = []
heapq.heappush(heap,5)
heapq.heappop(heap)
print(heap)

#인덱스와 값 같이
for i,p in enumerate(priorities):

#배열에 하나라도 있는지
any(i > 3 for i in [1,2,3])
#조합
from itertools import permutations
op = [list(y) for y in permutations(op)]
from itertools import combinations
op = [list(y) for y in combinations(op)]

#정렬

#자료형
#중복 x
list(set(list1))
#해쉬
key_value = {}


#문자열 -> 배열 숫자로
import re
ex = re.split(r'(\D)',expression)

#배열 다시 저장
_ex = _ex[:tmp]+_ex[tmp+2:]
#배열 복사
_ex = ex[:] 
#90도 회전
def rotate_a_matrix_by_90_degree(a):
    n = len(a)
    m = len(a[0])
    result = [[0] * n for _ in range(m)]
    for i in range(n):
        for j in range(m):
            result[j][n - i - 1] = a[i][j]
    return result

#2진법
format(len(no_zero), 'b')
#2진 탐색

from bisect import bisect_left, bisect_right
def count_by_range(a,left_value,right_value):
    right_index = bisect_right(a, right_value)
    left_index = bisect_left(a,left_value)
    return right_index - left_index

#중복 확인

from collections import deque
from itertools import combinations

def solution(relation):
    n_row = len(relation)
    n_col = len(relation[0])  

    candidates=[]
    for i in range(1,n_col+1):
        candidates.extend(combinations(range(n_col),i))
    final=[]
    for keys in candidates:
        tmp=[tuple([item[key] for key in keys]) for item in relation]
        
        if len(set(tmp))==n_row:
            final.append(keys)
    print(final)
    answer=set(final[:])
    for i in range(len(final)):
        for j in range(i+1,len(final)):
            if len(final[i])==len(set(final[i]).intersection(set(final[j]))):
                answer.discard(final[j])       
    return len(answer)
  

print(solution([["100","ryan","music","2"],["200","apeach","math","2"],["300","tube","computer","3"],["400","con","computer","4"],["500","muzi","music","3"],["600","apeach","music","2"]]))




#이진 트리 순회
import sys
sys.setrecursionlimit(10**6)


class Tree:
  def __init__(self,dataList):
    self.data = max(dataList,key=lambda x : x[1])
    leftList = list(filter(lambda x : x[0] < self.data[0],dataList))
    rightList = list(filter(lambda x : x[0] > self.data[0],dataList))
    if leftList != []:
      self.left = Tree(leftList)
    else:
      self.left = None
    
    if rightList !=[]:
      self.right = Tree(rightList)
    else:
      self.right = None

def fix(node,postList,preList):
  postList.append(node.data)
  if node.left is not None:
    fix(node.left,postList,preList)
  
  if node.right is not None:
    fix(node.right,postList,preList)
  
  preList.append(node.data)


def solution(nodeinfo):
  answer = []
  root = Tree(nodeinfo)
  postList=[]
  preList = []
  fix(root,postList,preList)
  print(list(preList))
  answer.append(list(map(lambda x: nodeinfo.index(x)+1 ,postList)))
  answer.append(list(map(lambda x: nodeinfo.index(x)+1 ,preList)))
 
  return answer

solution([[5,3],[11,5],[13,3],[3,5],[6,1],[1,3],[8,6],[7,2],[2,2]])




#정규표현식
re.findall('\d+', s)
import re


def solution(word,pages):
  urlToIdx = {}
  urlToScore = {}
  urlToExlink = {}
  word = word.lower()
  for i in range(len(pages)):
    lp = pages[i].lower()
    url = re.search(r'<meta[^>]*content="https://([\S]*)"/>',lp).group(1)
    urlToIdx[url] = i
    wordCnt = 0
    for find in re.findall(r'[a-zA-Z]+',lp):
      if find == word:
        wordCnt +=1
    s = set()

    for e in re.findall(r'<a href="https://[\S]*">',lp):
      s.add(re.search(r'"https://([\S]*)"',e).group(1))
    s=list(s)

    urlToScore[url] = list()
    urlToScore[url].append(wordCnt)
    urlToScore[url].append(len(s))

    for e in s :
      if e not in urlToExlink:
        urlToExlink[e] = list()
      urlToExlink[e].append(url)
  result = []
  for k,v in urlToScore.items():
    score = v[0]
    if k in urlToExlink:
      for u in urlToExlink[k]:
        score += urlToScore[u][0] / urlToScore[u][1]
    result.append([score,urlToIdx[k]])
  return sorted(result,key=lambda x:[-x[0],x[1]])[0][1]

print(solution('blind',["<html lang=\"ko\" xml:lang=\"ko\" xmlns=\"http://www.w3.org/1999/xhtml\">\n<head>\n  <meta charset=\"utf-8\">\n  <meta property=\"og:url\" content=\"https://a.com\"/>\n</head>  \n<body>\nBlind Lorem Blind ipsum dolor Blind test sit amet, consectetur adipiscing elit. \n<a href=\"https://b.com\"> Link to b </a>\n</body>\n</html>", "<html lang=\"ko\" xml:lang=\"ko\" xmlns=\"http://www.w3.org/1999/xhtml\">\n<head>\n  <meta charset=\"utf-8\">\n  <meta property=\"og:url\" content=\"https://b.com\"/>\n</head>  \n<body>\nSuspendisse potenti. Vivamus venenatis tellus non turpis bibendum, \n<a href=\"https://a.com\"> Link to a </a>\nblind sed congue urna varius. Suspendisse feugiat nisl ligula, quis malesuada felis hendrerit ut.\n<a href=\"https://c.com\"> Link to c </a>\n</body>\n</html>", "<html lang=\"ko\" xml:lang=\"ko\" xmlns=\"http://www.w3.org/1999/xhtml\">\n<head>\n  <meta charset=\"utf-8\">\n  <meta property=\"og:url\" content=\"https://c.com\"/>\n</head>  \n<body>\nUt condimentum urna at felis sodales rutrum. Sed dapibus cursus diam, non interdum nulla tempor nec. Phasellus rutrum enim at orci consectetu blind\n<a href=\"https://a.com\"> Link to a </a>\n</body>\n</html>"]
))



#테트리스 블록 없애기

#채워진 블록 모양 확인
def check_shape(start,color,row_num,board):
  y,x = start
  color = board[y][x]
  coord = set()
  color_cnt, black_cnt = 0,0
  
  #채워진 블록 모양 확인
  #[1,#]  [#,1]
  #[1,#]  [#,1]
  #[1,1]  [1,1]
  if row_num == 2 and y-2 >=0:
    for cx in range(x , x-2,-1 ):
      for cy in range(y, y-3,-1):
        if cy >= 0 and (board[cy][cx] == color or board[cy][cx] == '#'):
          if board[cy][cx] == color : color_cnt += 1
          if board[cy][cx] == '#' : black_cnt += 1
          coord.add((cy,cx))
        else:
          return None
    if color_cnt == 4 and black_cnt == 2:
      return coord
    else:
      return None

    #채워진 블록 모양 확인
  #[1,#,#]  [#,1,#] [#,#,1]
  #[1,1,1]  [1,1,1] [1,1,1]
  if row_num == 3 and y-1 >=0:
    for cx in range(x , x-3,-1 ):
      for cy in range(y, y-2,-1):
        if cy >= 0 and (board[cy][cx] == color or board[cy][cx] == '#'):
          if board[cy][cx] == color : color_cnt += 1
          if board[cy][cx] == '#' : black_cnt += 1
          coord.add((cy,cx))
        else:
          return None
    if color_cnt == 4 and black_cnt == 2:
      return coord
    else:
      return None


#위부터 블록 채우기
def build(board):
  for x in range(len(board[0])):
    for y in range(len(board)):
      if board[y][x] == '#':
        continue
      if board[y][x] == 0:
        board[y][x] = '#'
      elif board[y][x] != 0:
        break;
#밑변이 2,3 인 데이터 찾기
def search_shape(board):
  for y in range(len(board)-1,-1,-1):
    consecutive = 0
    color = None
    for x in range(len(board[0])):
      if board[y][x] == 0:
        color = None
        consecutive = 0
        continue
      if type(board[y][x]) is int and board[y][x] != 0:
        if color == None:
          consecutive = 1
          color = board[y][x]
          continue
        if board[y][x] != color:
          color = board[y][x]
          consecutive = 1
          continue
        if color == board[y][x]:
          consecutive += 1
        if 2 <= consecutive <= 3:
          coord = check_shape((y,x),color,consecutive,board)
          if coord is not None:
            return coord
  return None
def solution(board):
  cnt = 0

  while True:

    build(board)

    coord = search_shape(board)

    if coord is None:
      break
    
    cnt += 1

    remove(board,coord)
 
  return cnt

def remove(board,coord):
  for y,x in coord:
    board[y][x] = 0
    
solution([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,4,0,0,0],[0,0,0,0,0,4,4,0,0,0],[0,0,0,0,3,0,4,0,0,0],[0,0,0,2,3,0,0,0,5,5],[1,2,2,2,3,3,0,0,0,5],[1,1,1,0,0,0,0,0,0,5]])


def solution(n, arr1, arr2):
    answer = []
      # 두 배열 한번에
    for i,j in zip(arr1,arr2):
      # 비트연산
        a12 = str(bin(i|j)[2:])
        #오른쪽정렬
        a12=a12.rjust(n,'0')
        a12=a12.replace('1','#')
        a12=a12.replace('0',' ')
        answer.append(a12)
    return answer


#문자열을 돌면서 값을 확인할때는 -1을 이용하자

def solution(dartResult):
    score = []
    n = ''
    for i in dartResult:
        if i.isnumeric():
            n += i
        elif i == 'S':
            score.append(int(n) ** 1)
        elif i == 'D':
            score.append(int(n) ** 2)
        elif i == 'T':
            score.append(int(n) ** 3)
        elif i == '*':
            if len(score) > 1:
                score[-2] *= 2
            score[-1] *= 2
        elif i == '#':
            score[-1] *= -1
    return sum(score)


print(solution('1D2S3T*'))


import re
import math

def solution(str1, str2):
    #for 안에 if 가능 
    str1 = [str1[i:i+2].lower() for i in range(0, len(str1)-1) if not re.findall('[^a-zA-Z]+', str1[i:i+2])]
    str2 = [str2[i:i+2].lower() for i in range(0, len(str2)-1) if not re.findall('[^a-zA-Z]+', str2[i:i+2])]


    gyo = set(str1) & set(str2)
    hap = set(str1) | set(str2)

    if len(hap) == 0 :
        return 65536
    
    gyo_sum = sum([min(str1.count(gg), str2.count(gg)) for gg in gyo])
    hap_sum = sum([max(str1.count(hh), str2.count(hh)) for hh in hap])

    return math.floor((gyo_sum/hap_sum)*65536)



class G:
    pending = set()
    bombed = 0
    board = []

#밑에부터 없앨경우 에러  맨 뒤부터 없앰
def bomb():
    for i, j in reversed(sorted(G.pending)):
        G.board[i].pop(j)
        G.bombed += 1
    G.pending = set()

# 없앨 데이터는 중복되지 않게 set
def traverse(i, j):
    if j >= len(G.board[i+1]) - 1:
        return

    if G.board[i][j] == G.board[i][j+1] == G.board[i+1][j] == G.board[i+1][j+1]:
        for x, y in [(i + 1, j + 1), (i + 1, j), (i, j + 1), (i, j)]:
            G.pending.add((x, y))


def solution(m, n, board):
  # 중간에 빈공간이 생길경우 옆으로 누우면 된다
    G.board = [[board[i][j] for i in reversed(range(m))] for j in range(n)]
    while True:
        for i in range(len(G.board) - 1):
            for j in range(len(G.board[i]) - 1):
                traverse(i, j)
        if not G.pending:
            break
        bomb()

    return G.bombed

print(solution(6,6,['TTTANT', 'RRFACC', 'RRRFCC', 'TRRRAA', 'TTMMMF', 'TMMTTJ']))


#LRU

def solution(cacheSize,reference):
    cache = []
    cost = 0
    reference = [i.lower() for i in reference]
    if cacheSize == 0 :
      return len(reference) * 5
    for ref in reference:
        if not ref in cache:
            cost += 5
            if len(cache) < cacheSize:
                cache.append(ref)
            else:
                cache.pop(0)
                cache.append(ref)
        else:
            cost +=1
            cache.pop(cache.index(ref))
            cache.append(ref)
    return cost

print(solution(	0,['Jeju', 'Pangyo', 'Seoul', 'NewYork', 'LA', 'Jeju', 'Pangyo', 'Seoul', 'NewYork', 'LA']))


#그룹으로 더하기
import collections


def solution(participant, completion):
    answer = collections.Counter(participant) + collections.Counter(completion)
    return answer

#개수세기
def solution(clothes):
    from collections import Counter
    from functools import reduce
    cnt = Counter([kind for name, kind in clothes])
    answer = reduce(lambda x, y: x*(y+1), cnt.values(), 1) - 1
    return answer

solution([["yellow_hat", "headgear"], ["blue_sunglasses", "eyewear"], ["green_turban", "headgear"]])

#map이용해서 돌기
def solution(N, A):
    # write your code in Python 3.6
    li = {i:0 for i in range(1,N+1)}
    max_sum = 0
    max_num = 0
    for key in A:
        if key == N+1:
            max_sum += max_num
            li.clear()
            max_num = 0
        else:
            if li.get(key) is None:
                li[key] = 1
            else:
                li[key] += 1
                
            max_num = max(max_num,li[key])
    
    answer = [max_sum] * N
    #li = sorted(li)
    for key,val in li.items():
        answer[key-1] += val 
    
    return answer

print(solution(5,[3,4,4,6,1,4,4]))