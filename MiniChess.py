import math
import copy
import time
import argparse

class MiniChess:
    def __init__(self):
        self.current_game_state = self.init_board()

    """
    Initialize the board

    Args:
        - None
    Returns:
        - state: A dictionary representing the state of the game
    """
    def init_board(self):
        state = {
                "board": 
                [['bK', 'bQ', 'bB', 'bN', '.'],
                ['.', '.', 'bp', 'bp', '.'],
                ['.', '.', '.', '.', '.'],
                ['.', 'wp', 'wp', '.', '.'],
                ['.', 'wN', 'wB', 'wQ', 'wK']],
                "turn": 'white',
                }
        return state

    """
    Prints the board
    
    Args:
        - game_state: Dictionary representing the current game state
    Returns:
        - None
    """
    def display_board(self, game_state):
        print()
        for i, row in enumerate(game_state["board"], start=1):
            print(str(6-i) + "  " + ' '.join(piece.rjust(3) for piece in row))
        print()
        print("     A   B   C   D   E")
        print()

    """
    Check if the move is valid    
    
    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move which we check the validity of ((start_row, start_col),(end_row, end_col))
    Returns:
        - boolean representing the validity of the move
    """
    def is_valid_move(self, game_state, move):
        # Check if move is in list of valid moves
        valid_moves = self.valid_moves(game_state)
        return move in valid_moves

    """
    Returns a list of valid moves

    Args:
        - game_state:   dictionary | Dictionary representing the current game state
    Returns:
        - valid moves:   list | A list of nested tuples corresponding to valid moves [((start_row, start_col),(end_row, end_col)),((start_row, start_col),(end_row, end_col))]
    """
    def valid_moves(self, game_state):
        # Return a list of all the valid moves.
        # Implement basic move validation
        # Check for out-of-bounds, correct turn, move legality, etc
        board=game_state["board"]
        turn=game_state["turn"]
        moves=[] #Keeps a list of all possible moves on the current board, this is recalculated each turn
        for row in range(5):
            for column in range(5):
                piece = board[row][column]
                if piece == '.':                                #No piece at this position on the board
                    continue
                if 'K' in piece and piece.startswith(turn[0]):  #KING
                    moves+=self.calculateKingMoves(game_state,row,column)
                if 'B' in piece and piece.startswith(turn[0]):  #BISHOP
                    moves+=self.calculateBishopMoves(game_state,row,column)
                if 'Q' in piece and piece.startswith(turn[0]):  #QUEEN
                    moves+=self.calculateQueenMoves(game_state,row,column)
                if 'N' in piece and piece.startswith(turn[0]):  #KNIGHT
                    moves+=self.calculateKnightMoves(game_state,row,column)
                if 'p' in piece and piece.startswith(turn[0]):  #PAWN
                    moves+=self.calculatePawnMoves(game_state,row,column)
        print("FOR VERIFICATION, DELET LATER - Valid moves:", moves) #DELETE LATER (just to print all the valid moves)
        return moves
    
    def calculateKingMoves(self, game_state, row, column):
        # The King moves 1 square in any direction.
        # Unlike regular chess, the king can move to positions where it is under attack.
        # Capturing the opponent’s king results in a win.  
        board = game_state["board"]
        possibleMoves = []
        columnLetters = ['A', 'B', 'C', 'D', 'E'] #To convert the array position to the actual letter on the board
        # The king can move in 8 directions (diagonal and orthogonal)
        directions = [
            (1, 0), 
            (1, 1),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, 0),
            (-1, 1),
            (-1, -1)
        ]
        for rowDirection, columnDirection in directions:
            new_row = row + rowDirection
            new_col = column + columnDirection
            if 0 <= new_row < 5 and 0 <= new_col < 5:  # board bounds
                if board[new_row][new_col][0] == game_state["turn"][0]: # cannot move in friendly pawn space
                    continue
                #either enemy pawn or simple move
                possibleMoves.append(columnLetters[column]  + str(5-row) + " " + columnLetters[new_col] + str(5-new_row))          
        return possibleMoves
    
    def calculateQueenMoves(self, game_state, row, column):
        #The Queen is able to move any number of squares in any direction 
        #(similar to the king, but not limited to a single space)
        board = game_state["board"]
        possibleMoves = []
        columnLetters = ['A', 'B', 'C', 'D', 'E'] #To convert the array position to the actual letter on the board
        #The Queen can move in 8 directions until the limit of the board OR another piece is reached
        directions = [
            (1, 0), 
            (1, 1),
            (1, -1),
            (0, 1),
            (0, -1),
            (-1, 0),
            (-1, 1),
            (-1, -1)
        ]
        for rowDirection, columnDirection in directions:
            new_row=row
            new_column=column
            while True:
                new_row+=rowDirection
                new_column+=columnDirection
                if not (0<=new_column<5 and 0<=new_row<5):
                    break
                #stop move if the Queen hits a piece in that direction (cannot jump over) the last possible move should be taking the piece
                if board[new_row][new_column] != '.':
                    if board[new_row][new_column][0]!=game_state["turn"][0]:
                        possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[new_column] + str(5-new_row)) 
                    break
                possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[new_column] + str(5-new_row))  
        return possibleMoves
    
    def calculatePawnMoves(self, game_state, row, column):
        #moves a single square forward and can capture pieces that are a single square diagonally in front of it. 
        #When reaching the other side of the board (row 5 for white and row 1 for black), the pawn is promoted to a Queen.
        possibleMoves= []
        board = game_state["board"]
        columnLetters = ['A', 'B', 'C', 'D', 'E']
        #The pawn can move in twi directions, based on the players colour (black or white)
        #Black moves downwards, white moves upwards
        direction = -1
        if game_state["turn"]=="white":
            direction = 1
        if 0<=row+direction<5:
            #Simple move
            possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[column] + str(5-row+direction))
        #Check if the pawn can capture, in which case it may move diagonally
        if board[row+direction][column+1]!='.' and board[row+direction][column+1][0]!=game_state["turn"][0]:
            possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[column+1] + str(5-row+direction))
        if board[row+direction][column-1]!='.' and board[row+direction][column-1][0]!=game_state["turn"][0]:
            possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[column-1] + str(5-row+direction))
        return possibleMoves
    
    def calculateBishopMoves(self, game_state, row, column):
        #The Bishop can only move diagonally any number of squares, in any direction.
        possibleMoves=[]
        board=game_state["board"]
        columnLetters = ['A', 'B', 'C', 'D', 'E']
        directions=[
            (1,1),
            (1,-1),
            (-1,1),
            (-1,-1)
        ]
        for rowDirection, columnDirection in directions:
            new_row=row
            new_column=column
            while True:
                new_row+=rowDirection
                new_column+=columnDirection
                if not (0<=new_column<5 and 0<=new_row<5):
                    break
                #Stop if piece is hit
                if board[new_row][new_column]!='.':
                    #capture opponent piece
                    if board[new_row][new_column][0]!=game_state["turn"][0]:
                        possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[new_column] + str(5-new_row))
                    break
                possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[new_column] + str(5-new_row))  
        return possibleMoves
    
    def calculateKnightMoves(self, game_state, row, column):
        # The Knight moves in an L-shape the same way as regular chess (2 squares in one direction and 1 in a perpendicular direction) in any direction.
        # It is the only piece capable of jumping over other pieces.
        possibleMoves = []
        board = game_state["board"]
        columnLetters = ['A', 'B', 'C', 'D', 'E']
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for rowDirection, columnDirection in knight_moves:
            new_row = row + rowDirection
            new_col = column + columnDirection
            if 0 <= new_row < 5 and 0 <= new_col < 5:  # board bounds
                if board[new_row][new_col] == '.' or board[new_row][new_col][0] != game_state["turn"][0]:  # empty or opponent's piece
                    possibleMoves.append(columnLetters[column] + str(5 - row) + " " + columnLetters[new_col] + str(5 - new_row))
        return possibleMoves
    """
    Modify to board to make a move

    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    Returns:
        - game_state:   dictionary | Dictionary representing the modified game state
    """
    def make_move(self, game_state, move):
        start = move[0]
        end = move[1]
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]
        game_state["board"][start_row][start_col] = '.'
        game_state["board"][end_row][end_col] = piece
        game_state["turn"] = "black" if game_state["turn"] == "white" else "white"

        return game_state

    """
    Parse the input string and modify it into board coordinates

    Args:
        - move: string representing a move "B2 B3"
    Returns:
        - (start, end)  tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    """
    def parse_input(self, move):
        try:
            start, end = move.split()
            start = (5-int(start[1]), ord(start[0].upper()) - ord('A'))
            end = (5-int(end[1]), ord(end[0].upper()) - ord('A'))
            return (start, end)
        except:
            return None

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """
    def play(self):
        print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")
        while True:
            self.display_board(self.current_game_state)
            move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
            if move.lower() == 'exit':
                print("Game exited.")
                exit(1)

            move = self.parse_input(move)
            if not move or not self.is_valid_move(self.current_game_state, move):
                print("Invalid move. Try again.")
                continue

            self.make_move(self.current_game_state, move)



if __name__ == "__main__":
    game = MiniChess()
    game.play()