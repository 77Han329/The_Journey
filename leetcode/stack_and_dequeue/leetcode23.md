# MyQueue: Implement Queue using Two Stacks

## 📌 思路
- 使用两个栈 `stack_in` 和 `stack_out` 来模拟队列。
- **核心思想**：
  - 入队时，把元素放到 `stack_in`。
  - 出队/取队首时，如果 `stack_out` 为空，则将 `stack_in` 中的元素全部倒入 `stack_out`，保证顺序。
  - `stack_out.pop()` 就能实现 **FIFO（先进先出）**。



## 🧩 代码实现
```python
class MyQueue:

    def __init__(self):
        self.stack_in = []      # 入栈，用于存放新元素
        self.stack_out = []     # 出栈，用于弹出队首元素

    def push(self, x: int) -> None:
        """入队：往 stack_in 里加元素"""
        self.stack_in.append(x)

    def pop(self) -> int:
        """出队：如果 stack_out 为空，就把 stack_in 倒过来"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out.pop()

    def peek(self) -> int:
        """查看队首：逻辑同 pop，但不删除元素"""
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        return self.stack_out[-1]

    def empty(self) -> bool:
        """判断队列是否为空"""
        return not self.stack_in and not self.stack_out