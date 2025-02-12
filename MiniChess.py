import math
import copy
import time
import argparse
import os

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
        return True

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
        return

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
        file.write(f"Player: {player}, Turn #{turn}, Action: {action}\n")
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
        if tie:
            file.write(f"{player} won in {turn} turns!\n")
        else:
            file.write(f"Game tied in {turn} turns.\n")
        file.close()

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """
    def play(self):
        print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")

        # Create output file (add params later)
        output_file = self.create_game_trace_file("5", "100", "H", "H", False, self.current_game_state, None)

        while True:
            self.display_board(self.current_game_state)
            move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
            if move.lower() == 'exit':
                print("Game exited.")
                # add param to log
                self.log_end(output_file, "1", True, None)
                exit(1)

            move = self.parse_input(move)
            if not move or not self.is_valid_move(self.current_game_state, move):
                print("Invalid move. Try again.")
                continue

            self.make_move(self.current_game_state, move)
            # add params to log
            self.log_action(output_file, "1", "White", move, "H", self.current_game_state, None, None, None)

        # Log game is over
        # self.log_end(output_file, "1", False, "White")