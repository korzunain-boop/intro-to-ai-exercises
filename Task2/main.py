from two_player_games .games.dots_and_boxes import DotsAndBoxes, Player
import random
from datetime import datetime
from matplotlib import pyplot as plt

max_player = Player('1')
min_player = Player('2')


def evaluate(state, max_player, min_player):
    scores = state.get_scores()
    return scores[max_player] - scores[min_player]

def alphabeta(state, depth, alpha, beta):
    best_moves = []
    player = state.get_current_player()

    if depth == 0 or state.is_finished():
        return evaluate(state, max_player, min_player), None
    
    best_moves = None
    moves = state.get_moves()

    if player.char == '1':
        max_eval = -1e5
        for move in moves:
            new_state = state.make_move(move)
            eval_score, _ = alphabeta(new_state, depth-1, alpha, beta)
            if eval_score == max_eval:
                best_moves.append(move)
            if eval_score > max_eval:
                best_moves = []
                max_eval = eval_score
                best_move = move
            alpha = max(max_eval, alpha)
            if beta <= alpha:
                break
        if len(best_moves) == 0:
            return max_eval, best_move
        else:
            return max_eval, best_moves[random.randint(0, len(best_moves)-1)]

    else:
        min_eval = 1e5        
        for move in moves:
            new_state = state.make_move(move)
            eval_score, _ = alphabeta(new_state, depth-1, alpha, beta)
            if eval_score == min_eval:
                best_moves.append(move)
            if eval_score < min_eval:
                best_moves = []
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        if len(best_moves) == 0:
            return min_eval, best_move 
        else:
            return min_eval, best_moves[random.randint(0, len(best_moves)-1)]


def simulate_game(depth_max, depth_min, size):
    start_time = float(datetime.now().timestamp())
    game = DotsAndBoxes(size, max_player, min_player)
    while not game.is_finished():
        if game.get_current_player().char == '1':
            score, move = alphabeta(game.state, depth_max, alpha=-1e5, beta=1e5)
        else:
            score, move = alphabeta(game.state, depth_min, alpha=-1e5, beta=1e5)
        game.make_move(move)
    scores_dict = game.state.get_scores()
    scores = [scores_dict[max_player], scores_dict[min_player]]
    finish_time = float(datetime.now().timestamp())
    return scores,finish_time - start_time 

repeats = 100
max_N_M = 7
first_player_wins = [0 for i in range(0, max_N_M+1)]

for i in range(0, max_N_M+1):
    for j in range(1, repeats+1):
        scores, _ = simulate_game(2+i, 2, 2)
        if scores[1] > scores[0]:
            first_player_wins[i] += 1/repeats


plt.plot(range(0, max_N_M+1), first_player_wins)
plt.xticks(range(0, max_N_M+1))
plt.xlabel("N-M")
plt.ylabel("wins with N depth, %")
plt.show()

