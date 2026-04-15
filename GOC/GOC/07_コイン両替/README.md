# コイン両替

- ??: GOC
- ??ID: 7
- ????: 動的計画法
- ???: Easy
- ??????: LeetCode 322

## ???
硬貨の種類（coins）と目標金額（amount）が与えられる。目標金額を作るのに必要な最小硬貨枚数を返せ。不可能なら-1を返せ。

## ?
```text
入力: coins=[1,5,6,9], amount=11
出力: 2  # 5+6=11
```

## ???
dp[i] = 金額iを作る最小枚数。dp[0]=0、dp[i] = min(dp[i-c]+1) for c in coins。

## ???
時間: O(amount × coins数)　空間: O(amount)
