N, M = map(int, input().split())
C = list(map(int, input().split()))
INF = float('inf')
dp = [INF] * (M + 1)
dp[0] = 0
for j in range(1, M + 1):
    for c in C:
        if j >= c and dp[j - c] != INF:
            dp[j] = min(dp[j], dp[j - c] + 1)
print(dp[M] if dp[M] != INF else -1)
