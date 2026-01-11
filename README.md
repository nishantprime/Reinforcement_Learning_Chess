# ♟️ Neural Chess: A Hybrid Reinforcement Learning Bot

**Neural Chess** is a deep reinforcement learning project that teaches a neural network to play chess from scratch. It evolves from a random mover into a strategic player by combining a **Value Network** (Intuition) with **Minimax Search** (Calculation).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## 🧠 How It Works

This project moves beyond standard tutorials by implementing a **Hybrid Architecture** used by modern engines (like a simplified AlphaZero/Stockfish hybrid):

1.  **The Brain (Intuition):** A 3-layer Multi-Layer Perceptron (MLP) trained via Reinforcement Learning. It looks at a board state and predicts a "Material Advantage" score (e.g., +1.5 for a Pawn advantage).
2.  **The Training (Experience):** The bot plays thousands of games against itself on **Google Colab (T4 GPU)**. It uses an Epsilon-Greedy strategy to explore random moves and learn which ones lead to capturing pieces or checkmate.
3.  **The Search (Calculation):** During actual gameplay, the raw neural network is wrapped in a **Minimax Algorithm (Depth 2)**. This allows the bot to "think ahead" and reject moves that look good to the brain but lead to immediate blunders (like hanging a Queen).

## 🚀 Features

* **Reinforcement Learning Loop:** Trains purely by playing against itself (Self-Play).
* **Mixed Reward System:**
    * Capturing Pieces: **+Material Value** (Pawn=1, Queen=9)
    * Checkmate: **+1.5** (Priority over material)
    * Stalling Penalty: **-0.002** (Discourages shuffling pieces aimlessly)
* **Hybrid Inference:** Combines the speed of a Neural Net with the accuracy of Minimax Search.
* **Cross-Platform Workflow:**
    * **Train** on the Cloud (Google Colab) for GPU speed (~250 games/min).
    * **Watch/Play** locally on your PC (Pygame) with high-quality visualization.

## 📦 Installation

1.  **Install dependencies:**
    You need Python installed. Then run:
    ```bash
    pip install torch torchvision pygame chess
    ```

## 🛠️ Usage

### 1. Training the Brain (Cloud Recommended)
Training is computationally expensive. It is recommended to use the provided **Colab Notebook** to train on a T4 GPU.

1.  Open the `.ipynb` (or paste the training script into Google Colab).
2.  Run the training loop.
3.  Wait until `Epsilon` (randomness) drops below **0.30**.
4.  Download the model file (e.g., `chess_mlp_hybrid.pth`) from your Google Drive.

### 2. Watching the Bot Play (Local)
Once you have a trained model file, place it in the project directory.

##🧠 Why No Minimax During Training?
You might wonder: If we use Minimax to play, why not use it to train?

**Speed**: Training requires millions of decisions. Adding a search (checking 900+ positions per move) would slow training down by ~1000x.

**Generalization**: We want the Neural Network to develop "Intuition" (Pattern Recognition). By forcing it to evaluate static boards without calculation, it learns to recognize dangerous patterns (like open files or pins) instantly.

##🤝 Contributing
Feel free to fork this project! Ideas for improvements:

**Implement MCTS (Monte Carlo Tree Search) like AlphaZero.

**Add a "Policy Head" to the network to predict moves, not just scores.

**Create a "Play vs Human" mode (currently it plays itself).
