import math

# 定义棋盘大小
BOARD_SIZE = 5

# 定义玩家和AI的棋子类型
PLAYER = 'X'
AI = 'O'
EMPTY = ' '

# 创建初始棋盘
def create_board():
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return board

# 检查棋盘上是否有空位置
def has_empty(board):
    for row in board:
        if EMPTY in row:
            return True
    return False

# 检查是否有玩家或AI获胜
def is_winner(board, player):
    # 检查行
    for row in board:
        if all(cell == player for cell in row):
            return True

    # 检查列
    for col in range(BOARD_SIZE):
        if all(board[row][col] == player for row in range(BOARD_SIZE)):
            return True

    # 检查对角线
    if all(board[i][i] == player for i in range(BOARD_SIZE)):
        return True
    if all(board[i][BOARD_SIZE - 1 - i] == player for i in range(BOARD_SIZE)):
        return True

    return False

# 评估当前棋盘的得分
def evaluate(board):
    if is_winner(board, AI):
        return 1
    if is_winner(board, PLAYER):
        return -1
    return 0

# 使用Alpha-Beta剪枝算法搜索最佳移动
def alpha_beta(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or not has_empty(board):
        return evaluate(board)

    if maximizing_player:
        max_eval = -math.inf
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == EMPTY:
                    board[row][col] = AI
                    eval = alpha_beta(board, depth - 1, alpha, beta, False)
                    board[row][col] = EMPTY
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = math.inf
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row][col] == EMPTY:
                    board[row][col] = PLAYER
                    eval = alpha_beta(board, depth - 1, alpha, beta, True)
                    board[row][col] = EMPTY
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval

# 执行AI的移动
def make_move(board):
    best_eval = -math.inf
    best_move = None
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == EMPTY:
                board[row][col] = AI
                eval = alpha_beta(board, 4, -math.inf, math.inf, False)
                board[row][col] = EMPTY
                if eval > best_eval:
                    best_eval = eval
                    best_move = (row, col)
    board[best_move[0]][best_move[1]] = AI

# 打印棋盘
def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

# 主循环
def main():
    board = create_board()
    print("Welcome to Tic-Tac-Toe!")
    print("You are playing as 'X'.")
    print_board(board)

    while has_empty(board) and not is_winner(board, PLAYER) and not is_winner(board, AI):
        # 玩家下棋
        row = int(input("Enter the row (0-4): "))
        col = int(input("Enter the column (0-4): "))
        if board[row][col] == EMPTY:
            board[row][col] = PLAYER
            print_board(board)
        else:
            print("Invalid move! Try again.")
            continue

        if not has_empty(board) or is_winner(board, PLAYER) or is_winner(board, AI):
            break

        # AI执行AI的移动
        make_move(board)
        print("AI's move:")
        print_board(board)

    if is_winner(board, PLAYER):
        print("Congratulations! You win!")
    elif is_winner(board, AI):
        print("AI wins!")
    else:
        print("It's a draw!")

# 运行主程序
if __name__ == '__main__':
    main()