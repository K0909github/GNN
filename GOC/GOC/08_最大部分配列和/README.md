# 最大部分配列和

- ??: GOC
- ??ID: 8
- ????: 動的計画法
- ???: Easy
- ??????: LeetCode 53 (Kadane's Algorithm)

## ???
整数配列numsが与えられる。和が最大になる連続部分配列の和を返せ。

## ?
```text
入力: [-2,1,-3,4,-1,2,1,-5,4]
出力: 6  # [4,-1,2,1]
```

## ???
Kadane's Algorithm：現在の要素を「前の最大和に足す」か「そこから始め直す」かの選択。current = max(nums[i], current + nums[i])。

## ???
時間: O(N)　空間: O(1)
