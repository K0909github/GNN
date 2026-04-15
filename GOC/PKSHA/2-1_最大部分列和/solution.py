N = int(input())
A = list(map(int, input().split()))
cur = best = A[0]
for a in A[1:]:
    cur = max(a, cur + a)
    best = max(best, cur)
print(best)
