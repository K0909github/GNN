# コイン両替

- 出典: GOC
- 問題ID: 7
- カテゴリ: 動的計画法
- 難易度: Easy
- サブタイトル: LeetCode 322

## 問題文
硬貨の種類（coins）と目標金額（amount）が与えられる。目標金額を作るのに必要な最小硬貨枚数を返せ。不可能なら-1を返せ。

## 例
```text
入力: coins=[1,5,6,9], amount=11
出力: 2  # 5+6=11
```

## ヒント
dp[i] = 金額iを作る最小枚数。dp[0]=0、dp[i] = min(dp[i-c]+1) for c in coins。

## 計算量
時間: O(amount × coins数)　空間: O(amount)
