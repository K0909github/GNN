N, M = map(int, input().split())
P = sorted(map(int, input().split()))
count = total = 0
for p in P:
    if total + p <= M:
        total += p
        count += 1
    else:
        break
print(count)
