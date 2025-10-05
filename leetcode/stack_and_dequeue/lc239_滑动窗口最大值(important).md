# Sliding Window Maximum

🔗 **题目链接**: [LeetCode 239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

---

## 📌 题目描述
给定一个整数数组 `nums` 和一个窗口大小 `k`，请你输出每个滑动窗口中的最大值。

**示例**  
输入: nums = [1,3,-1,-3,5,3,6,7], k = 3
输出: [3,3,5,5,6,7]

---

## 💡 思路

1. **单调队列（Monotonic Queue）**  
   使用一个 **单调递减队列** 来保存窗口内的元素，保证队首元素始终是当前窗口的最大值。

2. **维护规则**  
   - 当新元素进入时，从队列尾部移除所有比它小的元素（这些元素未来不可能成为最大值）。  
   - 队首元素如果已经滑出窗口（即 `nums[i-k]`），则将它移除。  

3. **记录结果**  
   - 从第 `k-1` 个元素开始，每次窗口形成时，把队首元素（最大值）加入结果。  

---

## 📝 代码实现 (Python)

```python

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        kept_nums = deque()   # 单调递减队列
        res = []

        for i in range(len(nums)):
            # 维护kept nums
            self.maintain_deque(kept_nums, nums[i])

            # 移除已经滑出窗口的元素，然后如果要移除的数字是最大值，才pop 出去
            if i >= k and nums[i - k] == kept_nums[0]:
                kept_nums.popleft()

            # 记录窗口最大值
            if i >= k - 1:
                res.append(kept_nums[0])

        return res

    def maintain_deque(self, kept_nums: deque, num: int) -> None:
        #把小的数字放进去 
        while kept_nums and kept_nums[-1] < num:
            kept_nums.pop()
        kept_nums.append(num)