# COMP472_GroupProject_MiniChess
## Desciption
A simplified 5x5 chess game built in Python for a class project. The game currently supports Human vs Human mode and will later include Human vs AI and AI vs AI modes. The game generates an output file with detailed game traces, including board state, player actions, and heuristic information (when AI is enabled).  

## Features
- 5x5 Chessboard: Simplified chess pieces and moves  
- Game Modes: Human vs Human, Human vs AI (coming soon), AI vs AI (coming soon)  
- Alpha-Beta Pruning: Optionally enabled for AI players  
- Game Trace Output: Detailed game trace saved to the Downloads folder  

## Steps to run program
Step 1) Clone the repository: git clone https://github.com/COMP472_GroupProject_MiniChess.git  
Step 2) Install python (if not yet installed): Ensure you have Python 3.x installed  
Step 3) Navigate to the project folder: cd COMP472_GroupProject_MiniChess  
Step 4) Run the game: main.py  
Step 5) Set up game parameters (coming soon) 

## Added Classes:
- create_game_trace_file, log_action and log_end: these functions create the output file, and log what passes in the game
- isKingCaptured: Checks the board to see if either the black or white king is no longer present. Returns a string indicating which king is captured.
- promotePawn: Checks the first and last rows of the board for any pawns from the opposing side. If such pawns are present in those rows, they become queens.
- checkNumberOfPieces: Counts the current number of pieces at play on the board.
- calculateKingMoves: Returns a list of all the possible moves (in "B2 B3" format) of the king depending on its position (row and column in game_state["board"] ).
- calculateQueenMoves: Returns a list of all the possible moves (in "B2 B3" format) of a Queen depending on its position (row and column in game_state["board"] ).
- calculatePawnMoves: Returns a list of all the possible moves (in "B2 B3" format) of a pawn depending on its position (row and column in game_state["board"] ).
- calculateBishopMoves: Returns a list of all the possible moves (in "B2 B3" format) of the Bishop depending on its position (row and column in game_state["board"] ).
- calculateKnightMoves: Returns a list of all the possible moves (in "B2 B3" format) of the Knight depending on its position (row and column in game_state["board"] ).
