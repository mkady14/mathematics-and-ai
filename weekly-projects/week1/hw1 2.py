import random


def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 10)
        print(' ')


def check_winner(board, mark):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    return [mark, mark, mark] in win_conditions


def get_empty_positions(board):
    positions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                positions.append((i, j))
    return positions


def user_move(board):
    while True:
        try:
            row = int(input("Enter the row (1-3): ")) - 1
            col = int(input("Enter the column (1-3): ")) - 1
            if board[row][col] == " ":
                board[row][col] = "X"
                break
            else:
                print("This position is already taken.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter numbers between 1 and 3.")


def computer_move(board):
    empty_positions = get_empty_positions(board)

    if len(empty_positions) == 9:
        # if first move, start in the center
        move = (1, 1)

    else:
        # try to win or block user from winning
        move = None
        for i in empty_positions:
            # check if the computer can win
            board[i[0]][i[1]] = "O"
            if check_winner(board, "O"):
                move = i
                break
            board[i[0]][i[1]] = " "

        if move is None:
            for i in empty_positions:
                # check if the user can win and block them
                board[i[0]][i[1]] = "X"
                if check_winner(board, "X"):
                    move = i
                    break
                board[i[0]][i[1]] = " "

        if move is None:
            # make a random choice if there's no consequences
            move = random.choice(empty_positions)

    board[move[0]][move[1]] = "O"


def tic_tac_toe():
    board = [[" " for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic Tac Toe!")
    user_first = input("Do you want to go first? (y/n): ").lower() == 'y'

    for _ in range(9):
        print_board(board)
        if user_first:
            user_move(board)
            if check_winner(board, "X"):
                print_board(board)
                print("Congratulations! You win!")
                return
            user_first = False
        else:
            computer_move(board)
            if check_winner(board, "O"):
                print_board(board)
                print("Computer wins! Better luck next time.")
                return
            user_first = True

        if not get_empty_positions(board):
            print_board(board)
            print("It's a draw!")
            return


if __name__ == "__main__":
    tic_tac_toe()
