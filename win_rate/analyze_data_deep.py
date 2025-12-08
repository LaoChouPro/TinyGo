import os
import re
import random
import glob

def parse_last_stats(content):
    # Find all moves with comments
    # We want to capture: ;[BW][...]C[stats]
    # But regexing the whole file might be slow.
    # We really only care about the last few moves to verify the result.
    
    # Let's find all comments with the stats pattern
    # Pattern: C[0.52 0.48 0.00 0.6 ...]
    # We capture the color of the move associated with it if possible, but standard regex for "last match" is easier.
    
    # Iterate through all matches to check consistency
    # (Color, Val0, Val1, Val2)
    # We look for ;B[...]C[...] or ;W[...]C[...]
    # This is a bit complex for a single regex because [...] can contain anything.
    # However, standard SGF from KataGo is well formatted.
    
    # Let's just grab all stats and the color preceding them.
    # We assume the format is ;Color[...]C[Stats...]
    
    pattern = re.compile(r';([BW])\[[^\]]*\]\s*C\[\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)')
    matches = pattern.findall(content)
    
    return matches

def analyze_sgf_deep(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return None

    # Check Result
    re_match = re.search(r'RE\[([^\]]*)\]', content)
    result_str = re_match.group(1) if re_match else "Unknown"
    
    moves_data = parse_last_stats(content)
    if not moves_data:
        return None
        
    # Get last move info
    last_move_color, v0, v1, v2 = moves_data[-1]
    v0, v1, v2 = float(v0), float(v1), float(v2)
    
    return {
        'result': result_str,
        'last_move_color': last_move_color,
        'last_v0': v0, # Potential White Winrate
        'last_v1': v1, # Potential Black Winrate
        'last_v2': v2, # Draw/NoResult
        'num_moves': len(moves_data)
    }

data_dir = '/Users/laochou/Desktop/编程/项目/TinyGo-data/data'
# Use glob to get files (might be faster or just listdir)
files = [f for f in os.listdir(data_dir) if f.endswith('.sgf')]
# Sample 20000 files or all if less
sample_size = min(len(files), 20000)
sample_files = random.sample(files, sample_size)

print(f"Deep analyzing {sample_size} files...")

stats = {
    'processed': 0,
    'consistent_white_v0': 0, # v0 is high when White wins, low when Black wins
    'consistent_black_v0': 0, # v0 is high when Black wins (checking alternative hypothesis)
    'b_wins': 0,
    'w_wins': 0,
    'draws': 0,
    'v0_is_white_outliers': [], # Cases where Result says W+ but v0 < 0.5, or B+ but v0 > 0.5
    'sum_prob_error': 0, # Check if v0+v1+v2 ~ 1.0
    'turn_dependency_check': {'same': 0, 'diff': 0} # Check if values swap based on turn
}

# Threshold for "confident" win rate matching result
# If result is B+ and v0 < 0.1 -> Consistent with v0=White
# If result is W+ and v0 > 0.9 -> Consistent with v0=White
CONFIDENCE_THRESHOLD = 0.2 # 20% margin. 

for f in sample_files:
    info = analyze_sgf_deep(os.path.join(data_dir, f))
    if not info: continue
    
    stats['processed'] += 1
    
    res = info['result'].upper()
    v0, v1, v2 = info['last_v0'], info['last_v1'], info['last_v2']
    
    # Check probability sum
    if abs(v0 + v1 + v2 - 1.0) > 0.01:
        stats['sum_prob_error'] += 1

    # Determine Winner
    winner = None
    if res.startswith('B+'):
        winner = 'B'
        stats['b_wins'] += 1
    elif res.startswith('W+'):
        winner = 'W'
        stats['w_wins'] += 1
    else:
        winner = 'D'
        stats['draws'] += 1
        
    if winner == 'D':
        continue

    # Hypothesis: v0 is ALWAYS White Win Rate
    # If Winner is White, v0 should be high (> 0.5)
    # If Winner is Black, v0 should be low (< 0.5)
    is_v0_white_consistent = False
    if (winner == 'W' and v0 > 0.5) or (winner == 'B' and v0 < 0.5):
        stats['consistent_white_v0'] += 1
        is_v0_white_consistent = True
    else:
        # Outlier
        stats['v0_is_white_outliers'].append({
            'file': f,
            'result': res,
            'last_move': info['last_move_color'],
            'v0': v0,
            'v1': v1
        })

    # Check turn dependency
    # If v0 is White WR.
    # Does it matter if the last move was B or W?
    # We already checked consistency with Result.
    # Let's see if there's any correlation with who played.
    # Actually, if v0 is consistently White WR, then it is NOT turn dependent (in terms of index position).
    # If it were turn dependent (e.g. v0 = Current Player Win Rate), then:
    #   If Last Move = W (White just played), v0 (Current=White) should be High if W wins.
    #   If Last Move = B (Black just played), v0 (Current=Black) should be High if B wins.
    # Let's test "v0 is Current Player Win Rate" hypothesis briefly:
    #   Current Player is the one who *just played*? Or the one *about to play*?
    #   Usually comments correspond to the state *after* the move.
    #   So if ;B[...]C[...], the board state is after Black played. Next to play is White.
    #   If v0 is "Next Player Win Rate" (White), then it matches "v0 is White Win Rate".
    
    pass

print("Deep Analysis Results:")
print(f"Processed: {stats['processed']}")
print(f"B Wins: {stats['b_wins']}, W Wins: {stats['w_wins']}, Draws: {stats['draws']}")
print(f"Consistent with 'v0 = White Win Rate': {stats['consistent_white_v0']} ({stats['consistent_white_v0']/stats['processed']*100:.2f}%)")
print(f"Probability Sum Errors (>0.01): {stats['sum_prob_error']}")
print(f"Outliers (Result disagrees with v0=White): {len(stats['v0_is_white_outliers'])}")

if stats['v0_is_white_outliers']:
    print("\nSample Outliers:")
    for out in stats['v0_is_white_outliers'][:10]:
        print(out)
