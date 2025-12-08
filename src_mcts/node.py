import math
import numpy as np

class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {} # {action: node}
        self.prior = prior # P(s, a)
        
        self.visit_count = 0 # N
        self.value_sum = 0.0 # W
        
        self.is_expanded = False
        
    @property
    def value(self):
        # Q(s, a) = W / N
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def expand(self, action_probs):
        """
        action_probs: list of (action, prob)
        """
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prob)
        self.is_expanded = True
        
    def select_child(self, c_puct=1.0):
        """
        Select child using PUCT formula.
        PUCT = Q(s, a) + U(s, a)
        U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        
        Note: The value Q stored in a child node is from the perspective of the player
        who MOVED to get there. But we are at 'self' (current player).
        The child node represents the opponent's turn.
        Usually, child.value is "Win rate for Opponent".
        So for us, the value of that action is (1 - child.value).
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # N(s)
        sqrt_n = math.sqrt(self.visit_count)
        
        for action, child in self.children.items():
            # Q value for current player
            if child.visit_count > 0:
                # Assuming child.value is [0, 1] prob of Child's player winning.
                # Since Child's player is Opponent, our win prob is 1 - child.value.
                q_value = 1.0 - child.value
            else:
                # If unvisited, what is Q?
                # Usually assume loss or draw or use prior?
                # AlphaZero uses 0 (for -1 to 1 scale) or parent value.
                # Here we use 0.5 (unknown) or 0 (pessimistic).
                # Let's use 0.5 for [0,1] scale.
                q_value = 0.5
                
            u_value = c_puct * child.prior * sqrt_n / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
