import math
import copy
import time
import argparse
import os
import copy

# Basis for the tree needed to play minimax
class Node:
    def __init__(self, board=None, heurisitic=None, children=None, parent=None):
        self.board_state = board if board is not None else []
        self.heuristic = heurisitic
        self.children = children if children is not None else []
        self.parent = parent
        

    def addChild(self, child):
        self.children.append(child)

    def setParent(self, parent):
        self.parent = parent


class MiniChess:
    WHITE_KING_CAPTURED = 'wkc'
    BLACK_KING_CAPTURED = 'bkc'
    num_pieces = 12
    turn_counter = 1

    #Change these values to get best results
    MAX_DEPTH = 4 #Set to twice as much when alpha-beta is enabled
    MINIMAX_TIMEOUT_MULTIPLIER = 0.9 #Stop searching for a move after 90% of AI_TIMEOUT

    # Game parameters (set by user when game starts)
    MAX_TURNS = 0 #game ends after 100 turns
    ALPHA_BETA = False #set to false if we want to use simple minimax
    PLAYER_WHITE = "H"
    PLAYER_BLACK = "H"
    AI_TIMEOUT = 0 #AI must find a move in 5 seconds


    def __init__(self):
        self.current_game_state = self.init_board()
        # AI cumulative info
        self.total_states_explored = 0
        self.states_by_depth = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        self.total_branches = 0
        self.total_branch_nodes = 0

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
        if 0 <= row+direction < 5 and 0 <= column+1 < 5:  # Check boundaries
            if board[row+direction][column+1]!='.' and board[row+direction][column+1][0]!=game_state["turn"][0]:
                possibleMoves.append(columnLetters[column] + str(5-row) + " " + columnLetters[column+1] + str(5-row-direction))

        if 0 <= row+direction < 5 and 0 <= column-1 < 5:  # Check boundaries
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

    # Function to create the decision tree with all the game state nodes, setting children and parent
    def build_decision_tree(self, game_state, current_level, start_time):
        self.total_states_explored += 1
        if current_level in self.states_by_depth:
            self.states_by_depth[current_level] += 1
        else:
            self.states_by_depth[current_level] = 1

        # Create node given board state
        if (current_level == self.MAX_DEPTH) or (time.time() - start_time >= self.AI_TIMEOUT*self.MINIMAX_TIMEOUT_MULTIPLIER):  # Terminal node
            # Node gets a heuristic
            e = self.calculate_heuristic(game_state)
            board = copy.deepcopy(game_state["board"])
            return Node(board, e, [])
        else:
            # Node does not get a heuristic (will be calculated later during minimax)
            e = None
            board = copy.deepcopy(game_state["board"])
            current_node = Node(board, e, [])
            # Get valid moves and track branching factor
            valid_moves = self.valid_moves(game_state)
            #Branching factor
            if valid_moves:  # Only track non-leaf nodes
                self.total_branches += len(valid_moves)
                self.total_branch_nodes += 1
            
            # Add a node for each valid_move from this board_state
            for move in self.valid_moves(game_state):
                current_state = copy.deepcopy(game_state)
                new_state = self.make_move(current_state, self.parse_input(move))
                next_level = current_level + 1
                child_node = self.build_decision_tree(new_state, next_level,start_time)
                child_node.setParent(current_node)     
                current_node.addChild(child_node) 
            return current_node
        
    def minimax(self, node, depth, maximizingPlayer, start_time):   
        # Sometimes the heuristic is not calculated, so we calculate it here to avoid errors
        if node.heuristic is None:
            turn = "white" if maximizingPlayer else "black"
            node.heuristic = self.calculate_heuristic({"turn": turn, "board":node.board_state})

        if depth == 0 or not node.children:  #Terminal node (reached max depth or no more children)
            return node.heuristic, node #return the heuristic of the node aswell as the node to make the move
        
        if maximizingPlayer: #White is max
            #find the child node with the highest heuristic
            maxEval = -math.inf
            bestChild = None
            for child in node.children:
                eval = self.minimax(child, depth - 1, False, start_time)
                if eval[0] > maxEval:
                    maxEval = eval[0]
                    bestChild = child
            return maxEval, bestChild
        else: #Black is min
            #Find the child node with the lowest heuristic
            minEval = math.inf
            bestChild = None
            for child in node.children:
                eval = self.minimax(child, depth - 1, True, start_time)
                if eval[0] < minEval:
                    minEval = eval[0]
                    bestChild = child
            return minEval, bestChild

    
    def alphabeta(self, origin_node, depth, alpha, beta, maximizingPlayer):  #Depth is the same as maxdepth initially
        # Sometimes the heuristic is not calculated, so we calculate it here to avoid errors
        if origin_node.heuristic is None:
            origin_node.heuristic = self.calculate_heuristic(origin_node.board_state)

        if depth == 0 or not origin_node.children:  #Terminal node (reached max depth or no more children)
            return origin_node.heuristic, origin_node    
           
        # White is max
        # Black is min
        if maximizingPlayer:
            v = -math.inf
            best_node = None
            for child in origin_node.children:
                child_value, _ = self.alphabeta(child, depth - 1, alpha, beta, False)
                if child_value > v:
                    v = child_value
                    best_node = child
                alpha = max(alpha, v)
                if beta <= alpha: #Prune
                    break
            return v, best_node  # Return both the best value and the corresponding node
        else:
            v = math.inf
            best_node = None
            for child in origin_node.children:
                child_value, _ = self.alphabeta(child, depth - 1, alpha, beta, True)
                if child_value < v:
                    v = child_value
                    best_node = child
                beta = min(beta, v)
                if beta <= alpha: #Prune
                    break
            return v, best_node  # Return both the best value and the corresponding node

    # Utility function for turning string-format board coordinates into 2D-array indices
    def coordsToIndices(self, move_str):
        columnLetters = ['A', 'B', 'C', 'D', 'E']
        coordinates = str.split(move_str, ' ')[1]
        return [int(coordinates[1]) - 1, columnLetters.index(coordinates[0])]

    # Heuristic function e(n), which approximates the odds of White winning over Black
    # e1 increases the pawn's value based on how close it is to reaching the other side of the board (how close it is to becoming a queen)
    # e2 increases the piece's value based on whether or not it endangers the enemy king
    def calculate_heuristic(self, board_state, heuristic = "e2"):
        e1 = False 
        e2 = False
        if heuristic == "e1":
            e1 = True
        elif heuristic == "e2":
            e2 = True
        elif heuristic == "e0":
            e1 = False
            e2 = False
        else:
            print("Invalid heuristic '" + heuristic + "', please re-run the program with a valid heuristic option.")
            exit()

        e = 0
        E2_WEIGHT = 10

        # Current heurstic used for D2 is e_0 (see project spec)
        for row in range(0, len(board_state["board"])):
            for col in range(0, len(board_state["board"][row])):

                # e > 0: White has the advantage
                # e = 0: Game is in a neutral state
                # e < 0: Black has the advantage
                # Depending on the importance of the piece, their presence (or lack thereof) affects the odds for their respective side (pawns are worth 1, bishops and knights 3, etc.)
                if board_state["board"][row][col] == "wp":
                    if (e1):
                        # Calculate how close pawn is to opposing side, increase point value by inverse of total number of turns needed to reach it
                        e += 3 - row

                    # Determine pawn's possible moves. If the enemy king is at the end of one of those moves, increase pawn's weight
                    elif (e2):
                        for move in self.calculatePawnMoves(board_state, row, col):
                            r, c = self.coordsToIndices(move)
                            if (board_state["board"][r][c] == "bK"):
                                e += E2_WEIGHT
                                break

                    e += 1

                elif board_state["board"][row][col] == "wB" or board_state["board"][row][col] == "wN":
                    if (e2):
                        moves = self.calculateBishopMoves(board_state, row, col) if (board_state["board"][row][col] == "wB") else self.calculateKnightMoves(board_state, row, col)
                        for move in moves:
                            r, c = self.coordsToIndices(move)
                            if (board_state["board"][r][c] == "bK"):
                                e += 3 * E2_WEIGHT
                                break
                    e += 3

                elif board_state["board"][row][col] == "wQ":
                    if (e2):
                        for move in self.calculateQueenMoves(board_state, row, col):
                            r, c = self.coordsToIndices(move)
                            if (board_state["board"][r][c] == "bK"):
                                e += 9 * E2_WEIGHT
                                break
                    e += 9

                elif board_state["board"][row][col] == "wK":
                    if (e2):
                        for move in self.calculateKingMoves(board_state, row, col):
                            r, c = self.coordsToIndices(move)
                            if (board_state["board"][r][c] == "bK"):
                                e += E2_WEIGHT
                                break
                    e += 999
                    
                elif board_state["board"][row][col] == "bp":
                    if (e1):
                        # Calculate how close pawn is to opposing side, increase point value by inverse of total number of turns needed to reach it
                        e -= row - 1

                    # Determine pawn's possible moves. If the enemy king is at the end of one of those moves, increase pawn's weight
                    elif (e2):
                        for move in self.calculatePawnMoves(board_state, row, col):
                            r, c = self.coordsToIndices(move)
                            if (board_state["board"][r][c] == "wK"):
                                e -= E2_WEIGHT
                                break

                    e -= 1

                elif board_state["board"][row][col] == "bB" or board_state["board"][row][col] == "bN":
                    if (e2):
                        moves = self.calculateBishopMoves(board_state, row, col) if (board_state["board"][row][col] == "wB") else self.calculateKnightMoves(board_state, row, col)
                        for move in moves:
                            r, c = self.coordsToIndices(move)
                            if (board_state["board"][r][c] == "wK"):
                                e -= 3 * E2_WEIGHT
                                break
                    e -= 3

                elif board_state["board"][row][col] == "bQ":
                    if (e2):
                        for move in self.calculateQueenMoves(board_state, row, col):
                            r, c = self.coordsToIndices(move)
                            if (board_state["board"][r][c] == "wK"):
                                e -= 9 * E2_WEIGHT
                                break
                    e -= 9

                elif board_state["board"][row][col] == "bK":
                    if (e2):
                        for move in self.calculateKingMoves(board_state, row, col):
                            r, c = self.coordsToIndices(move)
                            if (board_state["board"][r][c] == "wK"):
                                e -= E2_WEIGHT
                                break

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
    def create_game_trace_file(self, timeout, max_turns, max_depth, player_1_type, player_2_type, alpha_beta, initial_state, heuristic=None):
        # Create output file in downloads folder
        downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        filename = os.path.join(downloads_folder, f"gameTrace-{str(alpha_beta).lower()}-{timeout}-{max_turns}.txt")
        file = open(filename, "w")

        # Write the game parameters
        file.write("=== Game Parameters ===\n")
        file.write(f"Timeout: {timeout} seconds\n")
        file.write(f"Max turns: {max_turns}\n")
        file.write(f"Max depth: {max_turns}\n")
        file.write(f"Player 1 type: {player_1_type} & Player 2 type: {player_2_type}\n")
        # AI parameters
        if player_1_type == "A" or player_2_type == "A":
            file.write(f"Alpha-Beta: {'ON' if alpha_beta else 'OFF'}\n")
            file.write(f"Heuristic used: {heuristic}\n")

        # Initial board configuration
        file.write("\n=== Initial Board Configuration ===\n")
        for i, row in enumerate(initial_state["board"], start=1):
            file.write(f"{6-i} {' '.join(piece.rjust(3) for piece in row)}\n")
        file.write("    A   B   C   D   E\n")
        
        # Game trace
        file.write("\n\n=== Game Trace ===\n")

        return file
    
    """
    Log action to output file

    Args:
        - File
        - Action information
        - New state
        - AI info
    Returns:
        - None
    """
    def log_action(self, file, turn, player, action, player_type, current_state, time_taken=None, heuristic_score=None, alpha_beta=False, method_score=None):
        file.write(f"\n\nPlayer: {player}, Type: {player_type}, Turn #{turn}, Action: {action}\n")
        if player_type == "A":
            file.write(f"Time for this action: {time_taken:.2f} sec\n")
            file.write(f"Heuristic score: {heuristic_score}\n")
            if (alpha_beta):
                file.write(f"Alpha-Beta search score: {method_score}\n")
            else:
                file.write(f"Min-Max search score: {method_score}\n")

        file.write("\n=== New Board Configuration ===\n")
        for i, row in enumerate(current_state["board"], start=1):
            file.write(f"{6-i} {' '.join(piece.rjust(3) for piece in row)}\n")
        file.write("    A   B   C   D   E\n")

        # Ai cumulative info
        if player_type == "A":
            file.write("\n=== AI Cumulative Information ===\n")
            # Format and display statistics
            file.write(f"Cumulative states explored: {self.format_number(self.total_states_explored)}\n")
            # Stats by depth
            depth_stats = ""
            depth_percentages = ""
            total = self.total_states_explored
            for depth, count in sorted({d:c for d,c in self.states_by_depth.items() if c > 0}.items()):
                depth_stats += f"{depth}={self.format_number(count)} "
                percentage = (count / total * 100) if total > 0 else 0
                if percentage >= 0.1:  # Only show depths with significant percentage
                    depth_percentages += f"{depth}={percentage:.1f}% "

            file.write(f"Cumulative states explored by depth: {depth_stats}\n")
            file.write(f"Cumulative % states explored by depth: {depth_percentages}\n")

            # Calculate average branching factor
            avg_branching = self.total_branches / self.total_branch_nodes if self.total_branch_nodes > 0 else 0
            file.write(f"Average branching factor: {avg_branching:.1f}\n")
    
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

    """
    Log human errors to output file

    Args:
        - parameters being used
    Returns:
        - None
    """
    def log_error(self, file, turn, player, action):
        file.write(f"\nInvalid input: {player} tried to enter {action} on turn {turn}.\n")

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """
    def play(self):
        print()
        self.user_game_parameters() # Get user input for game parameters
        # Create output file with parameters
        output_file = self.create_game_trace_file(self.AI_TIMEOUT, self.MAX_TURNS, self.MAX_DEPTH, self.PLAYER_WHITE, self.PLAYER_BLACK, self.ALPHA_BETA, self.current_game_state, "e0")

        # Play until a king is captured or we have a draw
        drawTimer = 20
        while self.isKingCaptured(self.current_game_state) == "" and drawTimer > 0 and self.turn_counter <= self.MAX_TURNS:
            # reset variables
            time_taken = None
            heuristic_score = None
            best_score = None

            self.display_board(self.current_game_state)
            current_player=self.current_game_state['turn'].capitalize()
            # track current player type
            current_player_ai = (self.current_game_state["turn"] == "white" and self.PLAYER_WHITE == "A") or (self.current_game_state["turn"] == "black" and self.PLAYER_BLACK == "A")
            

            # Human plays
            if (not current_player_ai):
                print("Enter moves as 'B2 B3'. Type 'exit' to quit.")
                move = input(f"{self.current_game_state['turn'].capitalize()} to move: ").strip()
                if move.lower() == 'exit':
                    print("Game exited.")
                    # Log game over
                    self.log_end(output_file, self.turn_counter, True)
                    exit(1)
                move_string = move # SAVE BEFORE CONVERSION FOR EASIER PRINTING
                move = self.parse_input(move)
                if not move or not self.is_valid_move(self.current_game_state, move):
                    print("Invalid move. Try again.")
                    self.log_error(output_file, self.turn_counter, current_player, move_string)
                    continue
                self.make_move(self.current_game_state, move)

            # AI plays
            elif (current_player_ai):
                print("Wait for AI to make a move...")
                start_time = time.time()
                # Run minimax or alphabeta
                if self.ALPHA_BETA:
                    # Create the decision tree and run alphabeta
                    decision_tree = self.build_decision_tree(self.current_game_state, 0, start_time)
                    best_score = self.alphabeta(decision_tree, self.MAX_DEPTH, -math.inf, math.inf, self.current_game_state["turn"] == "white")
                else:
                    # Create the decision tree and run minimax
                    decision_tree = self.build_decision_tree(self.current_game_state, 0,start_time)
                    best_score = self.minimax(decision_tree, self.MAX_DEPTH, self.current_game_state["turn"] == "white",start_time)
                # Get the best child node
                best_child = best_score[1]
                # Get the move that led to the best child node
                move = None
                for i in range(0, len(decision_tree.children)):               
                    if decision_tree.children[i] == best_child:
                        move = self.valid_moves(self.current_game_state)[i]
                        break
                # Make the move
                move_string = move  # Save string representation of the move
                heuristic_score = best_child.heuristic
                self.make_move(self.current_game_state, self.parse_input(move))
                time_taken = time.time() - start_time # Calculate time taken for AI move
                print(f"Time taken for AI move: {time_taken:.2f} sec")
                print(f"Heuristic score of adversarial search: {heuristic_score}")
                self.print_AI_info()  # Print cumulative AI info

            #POST TURN ACTIONS
            action_display = f"{current_player} moved from {move_string.split()[0]} to {move_string.split()[1]}"
            print()
            print(action_display)
            # Log action
            self.log_action(output_file, self.turn_counter, current_player, move_string.upper(), "A" if current_player_ai else "H", self.current_game_state, time_taken if time_taken is not None else None, heuristic_score if heuristic_score is not None else None, self.ALPHA_BETA, best_score[0] if best_score and len(best_score) > 0 and best_score[0] is not None else None)
            
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

        #GAME ENDS HERE
        print("Thank you for playing!")
        exit(1)


    def user_game_parameters(self):
        print("Welcome to Mini Chess! Type 'exit' to quit.")
        print("---Game Settings---")
        while True:
            self.PLAYER_WHITE = input("Player 1 - WHITE: Human (H) or AI (A)? ").strip().upper()
            if self.PLAYER_WHITE in {"H", "A"}:
                break
            print("Invalid input. Please enter 'H' for Human or 'A' for AI.")

        while True:
            self.PLAYER_BLACK = input("Player 2 - BLACK: Human (H) or AI (A)? ").strip().upper()
            if self.PLAYER_BLACK in {"H", "A"}:
                break
            print("Invalid input. Please enter 'H' for Human or 'A' for AI.")

        while True:
            user_input = input("Max turns before draw: ").strip()
            if user_input.isdigit() and int(user_input) > 0:
                self.MAX_TURNS = int(user_input)  # Convert to integer after validation
                break
            print("Invalid input. Please enter a positive integer.")

        if self.PLAYER_WHITE == "A" or self.PLAYER_BLACK == "A":
            while True:
                user_input = input("Timeout for AI moves (in seconds): ").strip()
                try:
                    user_value = float(user_input)  # Try converting input to float
                    if user_value >= 0:  # Ensure it's positive
                        self.AI_TIMEOUT = user_value
                        break
                    else:
                        print("Invalid input. Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a valid positive number.")

            while True:
                alpha_beta = input("Use Alpha-Beta pruning? (Y/N) ").strip().upper()
                if alpha_beta in {"Y", "N"}:
                    break
                print("Invalid input. Please enter 'Y' for Alpha-Beta or 'N' for Min-Max.")

            while True:
                user_input = input("Maximum depth: ").strip()
                if user_input.isdigit() and int(user_input) > 0:
                    self.MAX_DEPTH = int(user_input)  # Convert to integer after validation
                    break
                print("Invalid input. Please enter a positive integer.")

            if alpha_beta == "Y":
                self.ALPHA_BETA = True
                #self.MAX_DEPTH = self.MAX_DEPTH*2  # Deeper search with alpha-beta
            else:
                self.ALPHA_BETA = False

    def print_AI_info(self):
        print("=== AI Cumulative Information ===")
        # Format and display statistics
        print(f"Cumulative states explored: {self.format_number(self.total_states_explored)}")
        # Stats by depth
        depth_stats = ""
        depth_percentages = ""
        total = self.total_states_explored
        for depth, count in sorted({d:c for d,c in self.states_by_depth.items() if c > 0}.items()):
            depth_stats += f"{depth}={self.format_number(count)} "
            percentage = (count / total * 100) if total > 0 else 0
            if percentage >= 0.1:  # Only show depths with significant percentage
                depth_percentages += f"{depth}={percentage:.1f}% "

        print(f"Cumulative states explored by depth: {depth_stats}")
        print(f"Cumulative % states explored by depth: {depth_percentages}")

        # Calculate average branching factor
        avg_branching = self.total_branches / self.total_branch_nodes if self.total_branch_nodes > 0 else 0
        print(f"Average branching factor: {avg_branching:.1f}")

    # Helper method for number formatting
    def format_number(self, num):
        if num >= 1000000:
            return f"{num/1000000:.1f}M"
        elif num >= 1000:
            return f"{num/1000:.1f}k"
        else:
            return str(num)
