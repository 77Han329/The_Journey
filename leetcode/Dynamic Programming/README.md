# 动态规划

## 题目分类
1. 动态规划基础
2. 背包问题
3. 打家劫舍
4. 股票问题
5. 子序列问题


## 🧩 代码实现
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []

        for item in s:
            if item == '(':
                stack.append(')')
            elif item == '{':
                stack.append('}')
            elif item == '[':
                stack.append(']')
            else:
                # 遇到右括号时检查是否匹配
                # 情况1: stack 为空，证明不匹配
                # 情况2: 当我们的item 是） 但是栈顶不是） 那就是错了 
                if not stack or stack.pop() != item:
                    return False
        
        # 栈为空才是完全匹配
        
        return True if not stack else False
        