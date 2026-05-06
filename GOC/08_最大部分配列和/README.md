# 最大部分配列和

- 出典: GOC
- 問題ID: 8
- カテゴリ: 動的計画法
- 難易度: Easy
- サブタイトル: LeetCode 53 (Kadane's Algorithm)

## 問題文
整数配列numsが与えられる。和が最大になる連続部分配列の和を返せ。

## 例
```text
入力: [-2,1,-3,4,-1,2,1,-5,4]
出力: 6  # [4,-1,2,1]
```

## ヒント
Kadane's Algorithm：現在の要素を「前の最大和に足す」か「そこから始め直す」かの選択。current = max(nums[i], current + nums[i])。

## 計算量
時間: O(N)　空間: O(1)
