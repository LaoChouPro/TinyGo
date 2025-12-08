import os
import re
import random

def analyze_sgf(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return None

    # Check Size
    sz_match = re.search(r'SZ\[(\d+)\]', content)
    sz = int(sz_match.group(1)) if sz_match else 19 # Default to 19 if missing, but note it
    
    # Check Result
    re_match = re.search(r'RE\[([^\]]*)\]', content)
    result = re_match.group(1) if re_match else "Unknown"

    # Count moves
    # Simple heuristic: count ;B[ or ;W[
    moves = re.findall(r';[BW]\[', content)
    num_moves = len(moves)

    # Check Comments with winrate pattern
    # Pattern: 0.52 0.48 0.00 0.6
    # We look for C[float float float float
    comments_with_stats = re.findall(r'C\[\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.?\d*)', content)
    
    return {
        'file': os.path.basename(file_path),
        'sz': sz,
        'result': result,
        'num_moves': num_moves,
        'num_comments': len(comments_with_stats),
        'avg_winrate': float(comments_with_stats[-1][0]) if comments_with_stats else 0.5 # Last move winrate
    }

data_dir = '/Users/laochou/Desktop/编程/项目/TinyGo-data/data'
files = [f for f in os.listdir(data_dir) if f.endswith('.sgf')]
sample_files = random.sample(files, min(len(files), 1000))

stats = {
    'total': 0,
    'sz_19': 0,
    'has_result': 0,
    'fully_annotated': 0, # >90% moves have comments
    'b_wins': 0,
    'w_wins': 0,
    'draws': 0,
    'b_win_avg_last_wr': 0,
    'w_win_avg_last_wr': 0
}

b_win_count = 0
w_win_count = 0

print(f"Analyzing {len(sample_files)} files...")

for f in sample_files:
    info = analyze_sgf(os.path.join(data_dir, f))
    if not info: continue

    stats['total'] += 1
    if info['sz'] == 19: stats['sz_19'] += 1
    if info['result'] != "Unknown": stats['has_result'] += 1
    
    # Check annotation coverage
    if info['num_moves'] > 0 and info['num_comments'] / info['num_moves'] > 0.9:
        stats['fully_annotated'] += 1
        
    # Correlation
    res = info['result'].upper()
    last_wr = info['avg_winrate']
    
    if res.startswith('B+'):
        stats['b_wins'] += 1
        stats['b_win_avg_last_wr'] += last_wr
        b_win_count += 1
    elif res.startswith('W+'):
        stats['w_wins'] += 1
        stats['w_win_avg_last_wr'] += last_wr
        w_win_count += 1
    elif res == '0' or res == 'DRAW' or res == 'VOID':
        stats['draws'] += 1

if b_win_count > 0:
    stats['b_win_avg_last_wr'] /= b_win_count
if w_win_count > 0:
    stats['w_win_avg_last_wr'] /= w_win_count

print("Analysis Results:")
print(stats)
