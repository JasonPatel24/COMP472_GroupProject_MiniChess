import math
import copy
import time
import argparse

class MiniChess:
    WHITE_KING_CAPTURED = 'wkc'
    BLACK_KING_CAPTURED = 'bkc'
    num_pieces = 12

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
        move_str = f"{chr(move[0][1] + ord('A'))}{5 - move[0][0]} {chr(move[1][1] + ord('A'))}{5 - move[1][0]}"
        return move_str in valid_moves

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
        #The pawn can move in two directions, based on the players colour (black or white)
        #Black moves downwards, white moves upwards
        direction = 1   #Note: the direction is the value in the array!!
        if game_state["turn"]=="white":
            direction = -1
        if 0<=row+direction<5 and board[row + direction][column] == '.':
            #Simple move with no pieces in the way
            possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[column] + str(5-row-direction))
        #Check if the pawn can capture, in which case it may move diagonally
        if board[row+direction][column+1]!='.' and board[row+direction][column+1][0]!=game_state["turn"][0]:
            possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[column+1] + str(5-row-direction))
        if board[row+direction][column-1]!='.' and board[row+direction][column-1][0]!=game_state["turn"][0]:
            possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[column-1] + str(5-row-direction))
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
    
    def isKingCaptured(self, game_state):
        whiteKingCaptured = True
        blackKingCaptured = True

        for row in range(0, len(game_state["board"])):
            for col in range(0, len(game_state["board"][row])):
                # Check the board and verify whether the black or white king is still on the board
                if game_state["board"][row][col] == 'wK':
                    whiteKingCaptured = False

                if game_state["board"][row][col] == 'bK':
                    blackKingCaptured = False

        # Notify caller whether the black or white king has been captured, or say nothing if both kings are on the board
        if whiteKingCaptured:
            return self.WHITE_KING_CAPTURED
        
        elif blackKingCaptured:
            return self.BLACK_KING_CAPTURED
        
        else:
            return ""

    def promotePawn(self, game_state):
        # Promote the black pawns to queens if they're on white's side
        for i in range(0, len(game_state["board"][4])):
            if game_state["board"][4][i] == "bp":
                game_state["board"][4][i] = "bQ"
            

        # Promote the white pawns to queens if they're on black's side
        for i in range(0, len(game_state["board"][0])):
            if game_state["board"][0][i] == "wp":
                game_state["board"][0][i] = "wQ"

        # If no pawns have reached the opposite end of the board, nothing happens

    def checkNumberOfPieces(self, game_state):
        pieces = 0

        # Iterate over the board and see how many positions are occupied, this is the number of pieces on the board
        for row in range(0, len(game_state["board"])):
            for col in range(0, len(game_state["board"][row])):
                if game_state["board"][row][col] != '.':
                    pieces += 1

        return pieces

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
        
        # Play until a king is captured or we have a draw
        drawTimer = 20
        while self.isKingCaptured(self.current_game_state) == "" and drawTimer > 0:
            self.display_board(self.current_game_state)
            valid_moves = self.valid_moves(self.current_game_state)
            print(f"Valid moves for {self.current_game_state['turn']}: {valid_moves}")  # Print all valid moves
            move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
            if move.lower() == 'exit':
                print("Game exited.")
                exit(1)

            move = self.parse_input(move)
            if not move or not self.is_valid_move(self.current_game_state, move):
                print("Invalid move. Try again.")
                continue

            self.make_move(self.current_game_state, move)
            self.promotePawn(self.current_game_state) # Check if a pawn reached the other end of the board at the end of this turn, if so, promote it.
            
            # Check if game is moving towards a draw
            new_num_pieces = self.checkNumberOfPieces(self.current_game_state)
            if (self.num_pieces > new_num_pieces):
                # A piece has been captured
                drawTimer = 20
                self.num_pieces = new_num_pieces

            else:
                # No pieces have been captured this turn
                drawTimer -= 1

        # If game has ended, check which king is captured
        self.display_board(self.current_game_state)

        result = self.isKingCaptured(self.current_game_state)

        if result == self.WHITE_KING_CAPTURED:
            # Black wins
            print("Black wins!")

        elif result == self.BLACK_KING_CAPTURED:
            # White wins
            print("White wins!")

        else:
            # If no king is captured, then we have a draw
            print("Draw!")

        print("Thank you for playing!")
        exit(1)



if __name__ == "__main__":
    game = MiniChess()
    game.play()