# 🌳 Binary Tree Level Order Traversal (层序遍历)

**LeetCode题号**：102  
**题目链接**：[Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

---

## 🧩 题目描述

给定一棵二叉树的根节点 `root`，返回其节点值的 **层序遍历**（即逐层从左到右访问所有节点）。
## 💡 思路讲解

层序遍历（Level Order Traversal）属于 广度优先搜索（BFS）。

我们从根节点 root 开始，将其放入队列（deque）中。
然后每次循环取出当前层的所有节点，依次：
	1.	访问节点并记录其值；
	2.	将它的左、右子节点加入队列；
	3.	一层结束后，将该层的节点值列表加入结果数组中。

当队列为空时，说明所有节点都已遍历完毕。
## 🧠 关键点总结
	•	使用 队列（collections.deque） 实现 BFS；
	•	每一层的节点数量由 len(queue) 决定；
	•	每次循环处理完一整层节点；
	•	注意判空：if not root: return []；
	•	不需要递归，使用迭代即可。

## 🧱 代码实现

```python
输入：
    3
   / \
  9  20
    /  \
   15   7

输出：
[[3],[9,20],[15,7]]



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        from collections import deque 
        if not root:
            return []
        temp_nodes = deque()
        res = []

        
        temp_nodes.append(root)
        
        while (temp_nodes):
            size = len(temp_nodes)
            layer_res = []
            for _ in range(size):
                node_ = temp_nodes.popleft()
                layer_res.append(node_.val)

                if node_.left:
                    temp_nodes.append(node_.left)
                
                if node_.right:
                    temp_nodes.append(node_.right)
            res.append(layer_res)   
                
        return res