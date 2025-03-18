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
        # Create node given board state
        if (current_level == self.MAX_DEPTH) or (time.time() - start_time >= self.AI_TIMEOUT*self.MINIMAX_TIMEOUT_MULTIPLIER):  # Terminal node
            # Node gets a heuristic
            e = self.calculate_heuristic(game_state["board"])
            board = copy.deepcopy(game_state["board"])
            return Node(board, e, [])
        else:
            # Node does not get a heuristic (will be calculated later during minimax)
            e = None
            board = copy.deepcopy(game_state["board"])
            current_node = Node(board, e, [])
            
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
            node.heuristic = self.calculate_heuristic(node.board_state)

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
        if depth == 0 or not origin_node.children:  #Terminal node (reached max depth or no more children)
            return origin_node.heuristic, origin_node       
        # White is max
        # Black is min
        if maximizingPlayer:
            v = -math.inf
            for child in origin_node.children:
                v = max(v, self.alphabeta(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v
        else:
            v = math.inf
            for child in origin_node.children:
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
        return self.build_decision_tree(self.current_game_state, 0, self.MAX_DEPTH)

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
        output_file = self.create_game_trace_file(self.AI_TIMEOUT, self.MAX_TURNS, self.PLAYER_WHITE, self.PLAYER_BLACK, self.ALPHA_BETA, self.current_game_state)

        # Play until a king is captured or we have a draw
        drawTimer = 20
        while self.isKingCaptured(self.current_game_state) == "" and drawTimer > 0 and self.turn_counter <= self.MAX_TURNS:
            self.display_board(self.current_game_state)
            current_player=self.current_game_state['turn'].capitalize()
            # TODO: Remove these lines before submitting
            print()
            print(f"Current e(n): White advantage over Black: {self.calculate_heuristic(self.current_game_state['board'])}")

            # Human plays
            if (self.current_game_state["turn"] == "white" and self.PLAYER_WHITE == "H") or (self.current_game_state["turn"] == "black" and self.PLAYER_BLACK == "H"):
                print("Enter moves as 'B2 B3'. Type 'exit' to quit.")
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
                self.make_move(self.current_game_state, move)

            # AI plays
            elif (self.current_game_state["turn"] == "white" and self.PLAYER_WHITE == "A") or (self.current_game_state["turn"] == "black" and self.PLAYER_BLACK == "A"):
                print("Wait for AI to make a move...")
                start_time = time.time()
                # Run minimax or alphabeta
                if self.ALPHA_BETA:
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

            #POST TURN ACTIONS
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

        #GAME ENDS HERE
        print("Thank you for playing!")
        exit(1)


    def user_game_parameters(self):
        print("Welcome to Mini Chess! Type 'exit' to quit.")
        print("---Game Settings---")
        self.PLAYER_WHITE = input("Player 1 - WHITE: Human (H) or AI (A)? ").upper()
        self.PLAYER_BLACK = input("Player 2 - BLACK: Human (H) or AI (A)? ").upper()
        self.MAX_TURNS = int(input("Max turns before draw: "))
        if self.PLAYER_WHITE == "A" or self.PLAYER_BLACK == "A":
            self.AI_TIMEOUT = float(input("Timeout for AI moves (in seconds): "))
            alpha_beta = input("Use Alpha-Beta pruning? (Y/N) ").upper()
            if alpha_beta == "Y":
                self.ALPHA_BETA = True
                self.MAX_DEPTH = self.MAX_DEPTH*2  # Deeper search with alpha-beta
            else:
                self.ALPHA_BETA = False
