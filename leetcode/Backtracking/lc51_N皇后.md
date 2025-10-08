# N皇后

## 📖 题目链接
[LeetCode 51. NQueens](https://leetcode.com/problems/n-queens/)


## 🧩 代码实现
```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        board = ['.' * n for _ in range(n)]
        self.backtracking(0,board,n,res) 
        return res

    def backtracking(self,row,board,n,res):
        if row >= n:
            res.append(board[:])
        
        for col in range(0,n):
            if self.isValid(row,col,board):
                board[row] = board[row][:col] + 'Q' + board[row][col+1:]
                self.backtracking(row+1,board,n,res)
                board[row] =  board[row][:col] + '.' + board[row][col+1:]
                

    def isValid(self, row: int, col: int, chessboard: List[str]) -> bool:
       
        for i in range(row):
            if chessboard[i][col] == 'Q':
                return False 

        # 检查 45 度角是否有皇后
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if chessboard[i][j] == 'Q':
                return False  
            i -= 1
            j -= 1

        # 检查 135 度角是否有皇后
        i, j = row - 1, col + 1
        while i >= 0 and j < len(chessboard):
            if chessboard[i][j] == 'Q':
                return False  # 右上方向已经存在皇后，不合法
            i -= 1
            j += 1

        return True  # 当前位置合法
```