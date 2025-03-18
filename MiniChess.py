import math
import copy
import time
import argparse
import os
import copy

# Basis for the tree needed to play minimax
class Node:
    def __init__(self):
        self.parent = Node() #Parent Node of the current node, necessary for tracing back once the tree is built
        self.board_state = []
        self.heuristic = 0
        self.children = []

    def __init__(self, board, heurisitic, children):
        self.board_state = board
        self.heuristic = heurisitic
        self.children = children

    def addChild(self, child):
        self.children.append(child)


class MiniChess:
    WHITE_KING_CAPTURED = 'wkc'
    BLACK_KING_CAPTURED = 'bkc'
    num_pieces = 12
    turn_counter = 1

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
        turn_anouncement=f"Turn # {self.turn_counter}"
        print(turn_anouncement)
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

    # AI simulates all possible game states up after a number of moves equal to 'max_depth'. 
    def AI_play(self, game_state, current_level, max_depth):
        # White is max
        # Black is min

        # Create node given board state

        if (current_level == max_depth):
            # Node gets a heuristic
            e = self.calculate_heuristic(game_state["board"])
            board = copy.deepcopy(game_state["board"])
            return Node(board, e, [])

        else:
            # Node does not get a heuristic (will be calculated later during minimax)
            e = 0
            board = copy.deepcopy(game_state["board"])
            current_node = Node(board, e, [])
            
            # Add a node for each valid_move from this board_state
            for move in self.valid_moves(game_state):
                current_state = copy.deepcopy(game_state)
                new_state = self.make_move(current_state, self.parse_input(move))
                next_level = current_level + 1
                current_node.addChild(self.AI_play(new_state, next_level, max_depth))
            
            return current_node
        
    
    def alphabeta(self, node, depth, alpha, beta, maximizingPlayer):  #Depth is the same as maxdepth
        if depth == 0 or not node.children:  #Terminal node (reached max depth or no more children)
            return node.value
        if maximizingPlayer:
            v = -math.inf
            for child in node.children:
                v = max(v, self.alphabeta(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v
        else:
            v = math.inf
            for child in node.children:
                v = min(v, self.alphabeta(child, depth - 1, alpha, beta, True))
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v


    # Heuristic function e(n), which approximates the odds of White winning over Black
    def calculate_heuristic(self, board_state):
        e = 0

        # Current heurstic used for D2 is e_0 (see project spec)
        for row in range(0, len(board_state)):
            for col in range(0, len(board_state[row])):

                # e > 0: White has the advantage
                # e = 0: Game is in a neutral state
                # e < 0: Black has the advantage
                # Depending on the importance of the piece, their presence (or lack thereof) affects the odds for their respective side (pawns are worth 1, bishops and knights 3, etc.)
                if board_state[row][col] == "wp":
                    e += 1

                elif board_state[row][col] == "wB" or board_state[row][col] == "wN":
                    e += 3

                elif board_state[row][col] == "wQ":
                    e += 9

                elif board_state[row][col] == "wK":
                    e += 999
                    
                elif board_state[row][col] == "bp":
                    e -= 1

                elif board_state[row][col] == "bB" or board_state[row][col] == "bN":
                    e -= 3

                elif board_state[row][col] == "bQ":
                    e -= 9

                elif board_state[row][col] == "bK":
                    e -= 999

                else:
                    continue

        return e

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
    Create output file

    Args:
        - Parameters being used
        - Initial state
    Returns:
        - File used for output
    """
    def create_game_trace_file(self, timeout, max_turns, player_1_type, player_2_type, alpha_beta, initial_state, heuristic=None):
        # Create output file in downloads folder
        downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        filename = os.path.join(downloads_folder, f"gameTrace-{str(alpha_beta).lower()}-{timeout}-{max_turns}.txt")
        file = open(filename, "w")

        # Write the game parameters
        file.write("=== Game Parameters ===\n")
        file.write(f"Timeout: {timeout} seconds\n")
        file.write(f"Max turns: {max_turns}\n")
        file.write(f"Player 1: {player_1_type} & Player 2: {player_2_type}\n")
        # AI parameters
        if player_1_type == "AI" or player_2_type == "AI":
            file.write(f"Alpha-Beta: {'ON' if alpha_beta else 'OFF'}\n")
            file.write(f"Heuristic used: {heuristic}\n")

        # Initial board configuration
        file.write("\n=== Initial Board Configuration ===\n")
        for i, row in enumerate(initial_state["board"], start=1):
            file.write(f"{6-i} {' '.join(piece.rjust(3) for piece in row)}\n")
        file.write("    A   B   C   D   E\n")
        
        # Game trace
        file.write("\n=== Game Trace ===\n")

        return file
    
    """
    Log action to output file

    Args:
        - File
        - Action information
        - New state
    Returns:
        - None
    """
    def log_action(self, file, turn, player, action, player_type, current_state, time_taken=None, heuristic_score=None, alpha_beta_score=None):
        file.write(f"\nPlayer: {player}, Turn #{turn}, Action: {action}\n")
        if player_type == "AI":
            file.write(f"Time for this action: {time_taken:.2f} sec\n")
            file.write(f"Heuristic score: {heuristic_score}\n")
            file.write(f"Alpha-Beta search score: {alpha_beta_score}\n")

        file.write("=== New Board Configuration ===\n")
        for i, row in enumerate(current_state["board"], start=1):
            file.write(f"{6-i} {' '.join(piece.rjust(3) for piece in row)}\n")
        file.write("    A   B   C   D   E\n")

        # add cumulative ai info here once needed
    
    """
    Log game over to output file

    Args:
        - parameters being used
    Returns:
        - None
    """
    def log_end(self, file, turn, tie, player=None):
        if tie == False:
            file.write(f"\n{player} won in {turn} turns!\n")
        else:
            file.write(f"\nGame tied in {turn} turns.\n")
        file.close()

    # TODO: Remove function when done testing
    def testAIPlay(self):
        return self.AI_play(self.current_game_state, 0, 3)

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """
    def play(self):
        print()
        print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")
  
        # Create output file (add params later)
        output_file = self.create_game_trace_file("5", "100", "H", "H", False, self.current_game_state)
      
        # Play until a king is captured or we have a draw
        drawTimer = 20
        while self.isKingCaptured(self.current_game_state) == "" and drawTimer > 0:
            self.display_board(self.current_game_state)

            # TODO: Remove these lines before submitting
            print()
            print("Current e(n): White advantage over Black:")
            print(self.calculate_heuristic(self.current_game_state["board"]))

            valid_moves = self.valid_moves(self.current_game_state)
            print(f"Valid moves for {self.current_game_state['turn']}: {valid_moves}")  # Print all valid moves
            move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
            if move.lower() == 'exit':
                print("Game exited.")
                # Log game over
                self.log_end(output_file, self.turn_counter, True)
                exit(1)
            move_string = move # SAVE BEFORE CONVERSION FOR EASIER PRINTING
            move = self.parse_input(move)
            if not move or not self.is_valid_move(self.current_game_state, move):
                print("Invalid move. Try again.")
                continue
            current_player=self.current_game_state['turn'].capitalize() #SAVE BEFORE MAKING THE MOVE
            self.make_move(self.current_game_state, move)

            action_display = f"{current_player} moved from {move_string.split()[0]} to {move_string.split()[1]}"
            print()
            print(action_display)

            # Log action
            self.log_action(output_file, self.turn_counter, current_player, move_string.upper(), "H", self.current_game_state)
            
            if (self.current_game_state['turn'] == "white"):
                self.turn_counter+=1
            

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

        print()
        if result == self.WHITE_KING_CAPTURED:
            # Black wins
            print(f"Black wins in {self.turn_counter} turns!")
            # Log game over
            self.log_end(output_file, self.turn_counter, False, "Black")

        elif result == self.BLACK_KING_CAPTURED:
            # White wins
            print(f"White wins in {self.turn_counter} turns!")
            # Log game over
            self.log_end(output_file, self.turn_counter, False, "White")

        else:
            # If no king is captured, then we have a draw
            print(f"Draw after {self.turn_counter} turns!")
            # Log game over
            self.log_end(output_file, self.turn_counter, True)

        print("Thank you for playing!")
        exit(1)