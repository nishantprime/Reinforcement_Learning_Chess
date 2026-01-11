import pygame
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys  # <--- Added for command line args

# --- CONFIG ---
DEFAULT_MODEL_FILE = "chess_mlp_hybrid.pth"
WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEARCH_DEPTH = 2  # Depth 2 is standard for fast Python play

# --- 1. THE EYES (MATCHES COLAB) ---
def board_to_tensor(board):
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    tensor = torch.zeros(768, dtype=torch.float32, device=DEVICE)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            offset = piece_map[piece.piece_type]
            color = 0 if piece.color == chess.WHITE else 6
            idx = (offset + color) * 64 + i
            tensor[idx] = 1.0
    return tensor

# --- 2. THE BRAIN (MATCHES COLAB) ---
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))

# --- 3. MINIMAX ENGINE ---
def evaluate_position(board, model):
    # Immediate Game Over check
    if board.is_checkmate():
        if board.result() == "1-0": return 1000.0
        else: return -1000.0
    if board.is_game_over(): return 0.0

    # Neural Network Eval
    tensor = board_to_tensor(board)
    with torch.no_grad():
        score = model(tensor).item()
    return score * 10.0 # Scale for readability

def minimax(board, depth, alpha, beta, maximizing_player, model):
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model)

    legal_moves = list(board.legal_moves)
    
    if maximizing_player:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

def get_best_move(board, model):
    print(f"Thinking (Depth {SEARCH_DEPTH})...")
    
    legal_moves = list(board.legal_moves)
    if not legal_moves: return None

    best_move = None
    is_maximizing = (board.turn == chess.WHITE)
    best_value = -float('inf') if is_maximizing else float('inf')
    
    # Root Level Search
    for move in legal_moves:
        board.push(move)
        val = minimax(board, SEARCH_DEPTH - 1, -float('inf'), float('inf'), not is_maximizing, model)
        board.pop()
        
        if is_maximizing:
            if val > best_value:
                best_value = val
                best_move = move
        else:
            if val < best_value:
                best_value = val
                best_move = move
                
    print(f"Best Move: {best_move} | Eval: {best_value:.3f}")
    return best_move

# --- VISUALIZATION ---
def draw_board(screen, board):
    colors = [(238, 238, 210), (118, 150, 86)]
    for r in range(8):
        for c in range(8):
            color = colors[((r + c) % 2)]
            pygame.draw.rect(screen, color, pygame.Rect(c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    font = pygame.font.SysFont("segoeuisymbol", 50)
    piece_uni = {'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
                 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'}

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            symbol = piece_uni[piece.symbol()]
            text = font.render(symbol, True, (0, 0, 0))
            row, col = divmod(i, 8)
            row = 7 - row 
            text_rect = text.get_rect(center=(col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2))
            screen.blit(text, text_rect)
    pygame.display.flip()

def main():
    # --- COMMAND LINE ARGUMENT PARSING ---
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        model_file = DEFAULT_MODEL_FILE
        print(f"No model specified. Using default: {model_file}")
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Neural Minimax Chess ({model_file})")
    clock = pygame.time.Clock()
    
    # Load Model
    brain = ChessNet().to(DEVICE)
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file, map_location=DEVICE)
        brain.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded Brain! Trained on {checkpoint.get('games', '?')} games.")
    else:
        print(f"WARNING: Model '{model_file}' not found. Playing Randomly.")

    while True:
        board = chess.Board()
        running = True
        
        while not board.is_game_over() and running:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); return

            move = get_best_move(board, brain)
            if move:
                board.push(move)
            else:
                running = False 
                
            draw_board(screen, board)
            time.sleep(0.1) 

        print(f"Game Over. Result: {board.result()}")
        time.sleep(3) 

if __name__ == "__main__":
    main()