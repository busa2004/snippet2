def solution(s):
    def expand(left, right) :
        while left >= 0 and right <= len(s) and s[left] == s[right-1]:
            left -= 1
            right += 1
        return s[left+1:right-1]

    if len(s) < 2 or s == s[::-1]:
        return s
    result = ''
    for i in range(len(s) - 1):
        result = max(result, expand(i,i+1),expand(i,i+2),key=len)
    return result


# 느린 버전 모든 경우 다검사
def longest_palindrom(s):
    # 함수를 완성하세요
    return len(s) if s[::-1] == s else max(longest_palindrom(s[:-1]), longest_palindrom(s[1:]))