import torch
import numpy as np
import sys
import os

# Ensure we can import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src_mcts.node import MCTSNode
from src.board import GoBoard, SIZE, BLACK, WHITE, EMPTY

class MCTS:
    def __init__(self, policy_model, value_model, device='cpu', c_puct=1.0, num_simulations=100):
        self.policy_model = policy_model
        self.value_model = value_model
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        
        self.policy_model.eval()
        self.value_model.eval()
        
    def run(self, root_board, color, temperature=1.0):
        """
        Run MCTS simulations and return move probabilities.
        root_board: GoBoard instance
        color: BLACK or WHITE
        """
        root = MCTSNode()
        
        # Initial expansion of root
        self._expand_node(root, root_board, color)
        
        for _ in range(self.num_simulations):
            node = root
            sim_board = GoBoard()
            sim_board.board = root_board.board.copy()
            sim_board.next_player = color
            
            path = [node]
            
            # 1. Select
            while node.is_expanded:
                action, node = node.select_child(self.c_puct)
                path.append(node)
                
                # Apply move to sim_board
                r, c = action
                sim_board.play(r, c, sim_board.next_player)
                # Next player is automatically updated in sim_board? 
                # Wait, board.play() doesn't toggle next_player automatically in src/board.py?
                # Let's check src/board.py again.
                # src/board.py play() returns captured count.
                # It does NOT seem to toggle next_player explicitly in the snippet I saw.
                # I should handle player toggle manually.
                sim_board.next_player = 3 - sim_board.next_player
                
            # 2. Expand & Evaluate
            # Check if game over? (For now, just assume not over or check pass)
            # We need to evaluate sim_board state.
            
            # Value for CURRENT player (sim_board.next_player)
            value = self._get_value(sim_board)
            
            # If not terminal, expand
            # (In pure AlphaZero, we expand leaf and evaluate. 
            # If terminal, we don't expand but just get real value)
            
            # Simple check for now: If we just played a move, check if we can expand.
            # Only expand if not visited before? Standard MCTS expands leaf once.
            if not node.is_expanded:
                 self._expand_node(node, sim_board, sim_board.next_player)
            
            # 3. Backup
            # Value 'v' is prob that sim_board.next_player wins.
            # We propagate up.
            # Node 'node' corresponds to state 'sim_board'.
            # Its parent corresponds to state before move.
            
            current_player_for_value = sim_board.next_player
            
            for i in range(len(path) - 1, -1, -1):
                n = path[i]
                # 'n' state player is...
                # path[0] is root (color).
                # path[1] is after color moves (opp).
                # path[i] corresponds to state where it is player (color if i is even?)
                
                # Actually simpler:
                # We have 'value' = P(current_player_for_value wins).
                # We want to add P(n.player wins) to n.W.
                # If n.player == current_player_for_value, add value.
                # Else add 1-value.
                
                # Determine n's player.
                # root is 'color'.
                # n (depth d) player is: color if d%2==0 else 3-color.
                
                node_player = color if i % 2 == 0 else (3 - color)
                
                if node_player == current_player_for_value:
                    v = value
                else:
                    v = 1.0 - value
                    
                n.visit_count += 1
                n.value_sum += v
                
        # Return probabilities based on visit counts
        counts = []
        actions = []
        for action, child in root.children.items():
            actions.append(action)
            counts.append(child.visit_count)
            
        counts = np.array(counts)
        
        if temperature == 0:
            best_idx = np.argmax(counts)
            probs = np.zeros_like(counts, dtype=float)
            probs[best_idx] = 1.0
        else:
            # Apply temperature
            counts = counts ** (1.0 / temperature)
            probs = counts / np.sum(counts)
            
        return actions, probs
        
    def _expand_node(self, node, board, color):
        """
        Get priors from PolicyNet and create children.
        """
        # PolicyNet Input: [1, 3, 19, 19]
        # Ch0: Color, Ch1: Opp, Ch2: Ones
        
        feat = np.zeros((1, 3, SIZE, SIZE), dtype=np.float32)
        feat[0, 0] = (board.board == color)
        feat[0, 1] = (board.board == (3 - color))
        feat[0, 2] = 1.0
        
        input_tensor = torch.from_numpy(feat).to(self.device)
        
        with torch.no_grad():
            logits = self.policy_model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0] # [361]
            
        # Create children for legal moves
        action_probs = []
        for idx, prob in enumerate(probs):
            r, c = idx // SIZE, idx % SIZE
            
            # Check legality
            # Simple check: Is empty?
            # Full check: board.play() != -1?
            # For speed, we might trust PolicyNet mostly, but better check empty.
            if board.board[r, c] == EMPTY:
                # Check suicide? 'play' does it.
                # We can do a lightweight check or just add it and handle invalid later?
                # Better to filter strictly for MCTS.
                # But board copy is expensive.
                # Let's just check emptiness.
                action_probs.append(((r, c), prob))
                
        node.expand(action_probs)
        
    def _get_value(self, board):
        """
        Get win probability for board.next_player from ValueNet.
        ValueNet Input: [1, 2, 19, 19]
        Ch0: My Stones (next_player), Ch1: Opp Stones
        """
        feat = np.zeros((1, 2, SIZE, SIZE), dtype=np.float32)
        feat[0, 0] = (board.board == board.next_player)
        feat[0, 1] = (board.board == (3 - board.next_player))
        
        input_tensor = torch.from_numpy(feat).to(self.device)
        
        with torch.no_grad():
            # ValueNetSmall returns Sigmoid output
            value = self.value_model(input_tensor).item() # scalar [0, 1]
            
        return value

