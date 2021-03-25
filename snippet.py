#알파벳여부
'a'.isalpha()
#소문자변환
'A'.lower()
#큐
import collections
strs = collections.deque()
strs.append('a')
strs.popleft()
#정규식(숫자 문자만 남기기)
import re
s = 'a > 2 *'
s = re.sub('[^a-z0-9]','',s)
#문자열 슬라이싱(가장 빠른 속도)
s = 'abcde'
s[::1] #그냥 abcde
s[::-1] #반대로 edcba
s[::2] #두칸씩 acd
#lambda 정렬 인덱싱 슬라이싱을 통해 가능
['a b','c d'].sort(key=lambda x : (x.split()[1:],x.split()[0]))
['a b','c d'].sort(key=len) # 길이로 정렬

#배열에서 등장 횟수
import collections
collections.Counter(['tt','tt']).most_common(1) #몇개, 빈도수가 높은순서대로 [('tt',2)]
#dict 기본값 
import collections
s = ['eat','tea','tan','ate','nat','bat']
a = collections.defaultdict(list)
#.sort() 제자리 정렬 .sorted() return된 값 정렬
for word in s:
    a[''.join(sorted(word))].append(word)
#
import sys
mx = -sys.maxsize
mn = sys.maxsize