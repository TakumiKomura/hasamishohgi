import numpy as np
import pickle

from cell import Cell
from board import Board

class QLearningPlayer:
    def __init__(self, direction, epsilon=0.3, alpha=0.2, gamma=0.99):
        """
        :param direction: プレイヤーの向き(UP/DOWN)
        :param epsilon: 探索率（新しい手を試す確率）
        :param alpha: 学習率
        :param gamma: 割引率
        """
        self.direction = direction
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.last_state = None
        self.last_action = None
        
        self.load_q_table()

    def get_state_key(self, board):
        """盤面状態の表現"""
        # 盤面の状態のみを使用
        board_state = board.copy_state()
        
        # 状態をより簡潔に表現（シンメトリーを考慮）
        simplified_state = []
        for row in board_state:
            simplified_row = []
            for cell in row:
                if cell == self.direction:
                    simplified_row.append(1)  # 自分の駒
                elif cell == -self.direction:
                    simplified_row.append(-1)  # 相手の駒
                else:
                    simplified_row.append(0)  # 空マス
            simplified_state.append(tuple(simplified_row))
        
        return tuple(simplified_state)

    def get_action(self, board, possible_moves):
        """
        ε-greedy方策に基づいて行動を選択
        :param board: 現在の盤面
        :param possible_moves: 可能な手のリスト [(old_y, old_x, new_y, new_x), ...]
        :return: 選択された手
        """
        state = self.get_state_key(board)
        
        # 新しい状態の場合、Q値を初期化
        if state not in self.q_table:
            # 可能な手を文字列として保存
            self.q_table[state] = {str(move): 0.0 for move in possible_moves}

        # ε-greedy方策
        if np.random.random() < self.epsilon:
            # ランダムな行動を選択（探索）
            action = possible_moves[np.random.randint(len(possible_moves))]
            self.last_action = str(action)  # 文字列として保存
        else:
            # 最も価値の高い行動を選択（活用）
            self.last_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
            action = eval(self.last_action)  # 文字列をタプルに変換

        self.last_state = state
        return action  # タプルとして返す

    def learn(self, board, reward, done):
        """
        Q値を更新
        :param board: 新しい盤面
        :param reward: 報酬
        :param done: ゲーム終了フラグ
        """
        if self.last_state is None:
            return

        new_state = self.get_state_key(board)
        
        # 終了状態の場合
        if done:
            q_target = reward
        else:
            # 新しい状態のQ値の最大値を取得
            if new_state in self.q_table:
                max_future_q = max(self.q_table[new_state].values())
            else:
                max_future_q = 0
            q_target = reward + self.gamma * max_future_q

        # Q値の更新
        current_q = self.q_table[self.last_state][self.last_action]
        self.q_table[self.last_state][self.last_action] = current_q + self.alpha * (q_target - current_q)

    def save_q_table(self):
        """Q-tableを保存"""
        with open(f'q_table_{self.direction}.pkl', 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self):
        """保存されたQ-tableを読み込み"""
        try:
            with open(f'q_table_{self.direction}.pkl', 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}
