def maxSubArray(nums):
    current = best = nums[0]
    
    for n in nums[1:]:
        current = max(n, current + n)
        best = max(best, current)
    
    return best

print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))  # 6
print(maxSubArray([1]))                         # 1
print(maxSubArray([-1,-2,-3]))                  # -1
