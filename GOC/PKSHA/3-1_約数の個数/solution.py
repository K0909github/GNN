import math
N = int(input())
count = 0
i = 1
while i * i <= N:
    if N % i == 0:
        count += 2 if i * i != N else 1
    i += 1
print(count)
