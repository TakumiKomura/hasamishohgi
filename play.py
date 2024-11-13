import numpy as np
import pickle
import os

class Cell:
    # 駒の状態を表す定数
    UP = -1      # 上向きの駒
    DOWN = 1     # 下向きの駒
    EMPTY = 0    # 空のマス

    def __init__(self, condition):
        if condition == Cell.EMPTY:
            self.set_empty()
        else:
            self.set_condition(condition)
    
    def set_condition(self, condition):
        if condition != Cell.UP and condition != Cell.DOWN:
            raise Exception("Piece must be Piece.UP or Piece.Down.")
        self.condition = condition
    
    def set_empty(self):
        self.condition = Cell.EMPTY

class Board:
    AXIS_SIZE = 9    # ボードの大きさ（9x9）

    @staticmethod   
    def is_on_Board(y, x):
        """座標が盤面内かどうかを判定"""
        if y >= Board.AXIS_SIZE or y < 0:
            return False
        elif x >= Board.AXIS_SIZE or x < 0:
            return False
        return True

    def __init__(self) -> None:
        self.cells = [[Cell(Cell.EMPTY) for col in range(Board.AXIS_SIZE)] for row in range(Board.AXIS_SIZE)]
        self.cells[0] = [Cell(Cell.DOWN) for col in range(Board.AXIS_SIZE)]
        self.cells[Board.AXIS_SIZE - 1] = [Cell(Cell.UP) for col in range(Board.AXIS_SIZE)]

    def graphic_cell_condition(self, y, x):
        """セルの状態を視覚的な文字列で表現"""
        if self.cells[y][x].condition == Cell.DOWN:
            return f"\\_/"
        elif self.cells[y][x].condition == Cell.UP:
            return f"/^\\"
        return f"   "

    def show(self):
        print("   ", sep="", end="")
        for i in range(Board.AXIS_SIZE):
            print(f"  {i} ", sep="", end="")
        print()

        for i in range(Board.AXIS_SIZE):
            print("   -", sep="", end="")
            print("----"*Board.AXIS_SIZE, sep="")
            print(f" {i} ", sep="", end="")
            for j in range(Board.AXIS_SIZE):
                print(f"|{self.graphic_cell_condition(i, j)}", sep="", end="")
            print()
        print("   -", sep="", end="")
        print("----"*Board.AXIS_SIZE, sep="")
    
    def change_cell(self, old_y, old_x, new_y, new_x, direction):
        self.cells[old_y][old_x].set_empty()
        self.cells[new_y][new_x].set_condition(direction)

    def set_empty(self, y, x):
        self.cells[y][x].set_empty()

    def copy_state(self):
        """盤面の状態をコピ"""
        state = [[self.cells[y][x].condition for x in range(Board.AXIS_SIZE)] for y in range(Board.AXIS_SIZE)]
        return state

    def restore_state(self, state):
        """盤面の状態を復元"""
        for y in range(Board.AXIS_SIZE):
            for x in range(Board.AXIS_SIZE):
                self.cells[y][x].condition = state[y][x]

class Player:
    def __init__(self, direction, is_computer=False) -> None:
        """
        プレイヤーの初期化
        :param direction: プレイヤーの駒の向き(UP/DOWN)
        """
        self.direction = direction
        self.num_pieces = Board.AXIS_SIZE  # 所持している駒の数
        self.capture_count = 0             # 取った駒の数
        self.is_computer = is_computer  # 追加
        if direction == Cell.UP:
            self.str_direction = "UP"
        elif direction == Cell.DOWN:
            self.str_direction = "DOWN"

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

class Play:
    def __init__(self) -> None:
        self.board = Board()
        self.board.show()
        
        # プレイモードの選択
        mode = input("Select mode (1: Player vs Player, 2: Player vs Computer, 3: Computer vs Computer, 4: Q-Learning Training, 5: Player vs Q-Learning AI, 6: Minimax vs Q-Learning): ")
        if mode == "2":
            self.player1 = Player(Cell.UP)
            self.player2 = Player(Cell.DOWN, is_computer=True)
            try:
                self.delay = float(input("Enter delay between moves (seconds): "))
            except ValueError:
                self.delay = 1
        elif mode == "3":
            self.player1 = Player(Cell.UP, is_computer=True)
            self.player2 = Player(Cell.DOWN, is_computer=True)
            try:
                self.delay = float(input("Enter delay between moves (seconds): "))
            except ValueError:
                self.delay = 1
        elif mode == "4":
            # Q学習モード（訓練）
            self.player1 = Player(Cell.UP)
            self.player2 = Player(Cell.DOWN)
            self.player1.q_learning = QLearningPlayer(Cell.UP)
            self.player2.q_learning = QLearningPlayer(Cell.DOWN)
            
            # 学習モードの場合は自動再生の速度を設定
            try:
                self.delay = float(input("Enter delay between moves (seconds): "))
                self.num_episodes = int(input("Enter number of episodes to train: "))
            except ValueError:
                self.delay = 0.1
                self.num_episodes = 1000
        elif mode == "5":
            # Q学習AIとの対戦モード
            player_side = input("Choose your side (1: UP, 2: DOWN): ")
            if player_side == "1":
                self.player1 = Player(Cell.UP)
                self.player2 = Player(Cell.DOWN, is_computer=True)
                self.player2.q_learning = QLearningPlayer(Cell.DOWN, epsilon=0)  # epsilon=0で最適行動のみを選択
            else:
                self.player1 = Player(Cell.UP, is_computer=True)
                self.player1.q_learning = QLearningPlayer(Cell.UP, epsilon=0)
                self.player2 = Player(Cell.DOWN)
            self.delay = 0.5  # AIの思考時間
        elif mode == "6":
            # Minimax vs Q-Learning AIの対戦モード
            self.num_episodes = int(input("Enter number of episodes: "))
            self.delay = float(input("Enter delay between moves (seconds, 0 for no delay): "))
            self.train_minimax_vs_qlearning()
        else:
            self.player1 = Player(Cell.UP)
            self.player2 = Player(Cell.DOWN)
        
        self.turn = self.player1.direction
        self.difference_3_pieces = False
        self.is_finished = False
        self.move_history = []
        
        if mode == "4":
            self.train_q_learning()
        else:
            self.play()

    def change_turn(self):
        self.turn = -self.turn

    def play(self):
        while(not self.is_finished):
            self.play_one_turn()
    
    def play_one_turn(self):
        """1ターンの処理を実行"""
        # 入力処理 
        old_y, old_x, new_y, new_x = self._get_valid_move()

        # 駒の移動と取得処理
        self._execute_move(old_y, old_x, new_y, new_x)

        # 勝利判定
        if self._check_winner():
            self.show_captured_count()
            return

        # 次のターンの準備
        self.board.show()
        self.show_captured_count()
        self.change_turn()

    def _get_valid_move(self):
        """有効な移動入力を取得"""
        current_player = self.player1 if self.turn == self.player1.direction else self.player2
        print(f"turn:{current_player.str_direction}")

        # コンピュータの手番の場合
        if current_player.is_computer:
            return self._get_computer_move()

        # 人間プレイヤーの手番
        while True:
            try:
                # undo 処理
                command = input("Enter 'u' to undo, or press Enter to continue: ")
                if command.lower() == 'u':
                    if self.undo_move():
                        continue
                    else:
                        print("Cannot undo - no previous moves.")
                        continue

                # 移動元と移動先の座標を入力
                old_y = int(input("from y:"))
                old_x = int(input("from x:"))
                new_y = int(input("to y:"))
                new_x = int(input("to x:"))

                # 入力確認
                if input("retype? y or n :") == "n" and self.is_movable(old_y, old_x, new_y, new_x):
                    return old_y, old_x, new_y, new_x

                raise Exception("Invalid move command.")
            except Exception as e:
                print(e)
                print("Type again.")

    def _execute_move(self, old_y, old_x, new_y, new_x):
        """駒の移動と取得処理を実行"""
        # 現在の状態を保存
        current_state = {
            'board': self.board.copy_state(),
            'player1_pieces': self.player1.num_pieces,
            'player2_pieces': self.player2.num_pieces,
            'player1_capture': self.player1.capture_count,
            'player2_capture': self.player2.capture_count,
            'difference_3_pieces': self.difference_3_pieces
        }
        self.move_history.append(current_state)

        self.board.change_cell(old_y, old_x, new_y, new_x, self.turn)
        self.capture_pieces(new_y, new_x, self.turn)

        # 3枚差ルールの処理
        captured_count = len(self.get_capturable_pieces(new_y, new_x, self.turn))
        if self.difference_3_pieces and captured_count > 0:
            self.difference_3_pieces = False

    def _check_winner(self):
        """勝者を判定して結果を表示"""
        winner_direction = self.get_won_player()
        if winner_direction is not None:
            self.is_finished = True
            # 勝者のプレイヤーを特定
            winning_player = self.player1 if winner_direction == self.player1.direction else self.player2
            winner = "Q-Learning" if hasattr(winning_player, 'q_learning') else "Minimax"
            print(f"Winner: {winner}")
            return True
        return False

    def is_movable(self, old_y, old_x, new_y, new_x):
        # 移動元と移動先が範囲外でないかを確認
        if  not Board.is_on_Board(old_y, old_x) or not Board.is_on_Board(new_y, new_x):
            return False
        
        # 移動元のセルを取得
        old_position = self.board.cells[old_y][old_x]

        # 移動元のセルがコマであるか確認
        if old_position.condition == Cell.EMPTY:
            return False
        
        # 移動元と移動先が同じでないかを確認
        if old_y == new_y and old_x == new_x:
            return False
        
        # 水平方向または垂直方向の移動かを確認
        if old_y != new_y and old_x != new_x:
            return False

        # 移動経路上に他のコマがないか認
        if old_y == new_y:  # 水平方向の移動
            step = 1 if old_x < new_x else -1
            for col in range(old_x + step, new_x + step, step):
                if self.board.cells[new_y][col].condition != Cell.EMPTY:
                    return False
        elif old_x == new_x:  # 垂直方向の移動
            step = 1 if old_y < new_y else -1
            for row in range(old_y + step, new_y + step, step):
                if self.board.cells[row][new_x].condition != Cell.EMPTY:
                    return False

        return True
    
    def capture_piece(self, y, x, direction):
        self.board.set_empty(y, x)
        if direction == self.player1.direction:
            self.player1.num_pieces -= 1
            self.player2.capture_count += 1
        else:
            self.player2.num_pieces -= 1
            self.player1.capture_count += 1
    
    def capture_pieces(self, y, x, current_turn):
        capturable_pieces = self.get_capturable_pieces(y, x, current_turn)
        for piece in capturable_pieces:
            self.capture_piece(piece[0], piece[1], -current_turn)

    def get_capturable_pieces(self, y, x, current_turn):
        capturable_pieces = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dy, dx)

        for dy, dx in directions:
            temp_y, temp_x = y + dy, x + dx
            
            # 盤面外または空のマスの場合はスキップ
            if not Board.is_on_Board(temp_y, temp_x) or self.board.cells[temp_y][temp_x].condition == Cell.EMPTY:
                continue
            
            # 隣接するマスが相手のコマの場合、連結領域を探索
            if self.board.cells[temp_y][temp_x].condition == -current_turn:
                connected_region = self.get_connected_region(temp_y, temp_x, -current_turn)
                
                # 連結領域が有効で、かつ取れる場合
                if connected_region and self.is_region_capturable(connected_region, current_turn):
                    capturable_pieces.extend(connected_region)
            
            count = 1
            # 通常のはさみの確認
            while Board.is_on_Board(temp_y, temp_x):
                current_cell = self.board.cells[temp_y][temp_x]
                
                # 空のマスがある場合は終了
                if current_cell.condition == Cell.EMPTY:
                    break
                    
                # 自分の駒で挟めた場合
                if current_cell.condition == current_turn:
                    # 間の駒を killable_pieces に追加
                    for i in range(1, count):
                        kill_y = y + (dy * i)
                        kill_x = x + (dx * i)
                        capturable_pieces.append((kill_y, kill_x))
                    break
                    
                count += 1
                temp_y += dy
                temp_x += dx
        
        return capturable_pieces

    def get_connected_region(self, start_y, start_x, piece_type):
        """指定された位置から同じ種類のコマの連結領を返す"""
        
        connected = set()
        to_check = [(start_y, start_x)]
        while to_check:
            y, x = to_check.pop()
            if (y, x) in connected:
                continue
                
            connected.add((y, x))
            
            # 隣接するマスを確認
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_y, new_x = y + dy, x + dx
                if (Board.is_on_Board(new_y, new_x) and 
                    self.board.cells[new_y][new_x].condition == piece_type and 
                    (new_y, new_x) not in connected):
                    to_check.append((new_y, new_x))
        
        return list(connected)

    def is_region_capturable(self, region, current_turn):
        """連結領域が取れるかどうかを判定"""

        surrounding = set() # 連結領域の周囲のマス
        for y, x in region:
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_y, new_x = y + dy, x + dx
                if not Board.is_on_Board(new_y, new_x):
                    continue
                if (new_y, new_x) not in region:
                    surrounding.add((new_y, new_x))
        
        # 周囲のマスが全て自分のコマか盤面の端である必要がある
        for y, x in surrounding:
            if self.board.cells[y][x].condition != current_turn:
                return False
        
        return True
    
    def get_won_player(self):
        # 5枚以上取られた場合の判定
        if self.player1.capture_count >= 5:
            return self.player1.direction
        elif self.player2.capture_count >= 5:
            return self.player2.direction
        
        # 3枚差の判定(直後に相手がコマを取り返したら勝利しない)
        if self.player1.capture_count - self.player2.capture_count >= 3:
            if self.difference_3_pieces == True:
                return self.player1.direction
            else:
                self.difference_3_pieces = True
        elif self.player2.capture_count - self.player1.capture_count >= 3:
            if self.difference_3_pieces == True:
                return self.player2.direction
            else:
                self.difference_3_pieces = True
        
        return None
    
    def show_captured_count(self):
        print(f"captured count: {self.player1.str_direction}...{self.player1.capture_count}, {self.player2.str_direction}...{self.player2.capture_count}")

    def undo_move(self):
        """直前の手を取り消す"""
        if not self.move_history:
            return False

        # 前の状態を復元
        previous_state = self.move_history.pop()
        self.board.restore_state(previous_state['board'])
        self.player1.num_pieces = previous_state['player1_pieces']
        self.player2.num_pieces = previous_state['player2_pieces']
        self.player1.capture_count = previous_state['player1_capture']
        self.player2.capture_count = previous_state['player2_capture']
        self.difference_3_pieces = previous_state['difference_3_pieces']

        # ターンを戻す
        self.change_turn()
        
        # 盤面を表示
        self.board.show()
        self.show_captured_count()
        return True

    def _get_computer_move(self):
        """コンピュータの手を生成"""
        import time
        time.sleep(self.delay)

        current_player = self.player1 if self.turn == self.player1.direction else self.player2
        
        # Q学習AIが設定されている場合はそちらを使用
        if hasattr(current_player, 'q_learning'):
            return self._get_q_learning_move()
        
        # 通常のMinimax AIを使用
        return self._get_minimax_move()

    def _get_minimax_move(self):
        """Mini-Max法による手の生成"""
        def minimax(depth, is_maximizing, alpha, beta):
            if depth == 0:
                return None, self._evaluate_board()
            
            best_move = None
            if is_maximizing:
                max_eval = float('-inf')
                moves = self._get_possible_moves(self.turn)
                for move in moves:
                    old_y, old_x, new_y, new_x = move
                    # 手を実行
                    board_state = self.board.copy_state()
                    self.board.change_cell(old_y, old_x, new_y, new_x, self.turn)
                    
                    # 移動の評価値を計算
                    move_score = self.evaluate_move(old_y, old_x, new_y, new_x, self.turn)
                    
                    # 再帰的に評価
                    _, eval = minimax(depth - 1, False, alpha, beta)
                    eval += move_score  # 移動の評価値を加算
                    
                    # 状態を戻す
                    self.board.restore_state(board_state)
                    
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                return best_move, max_eval
            else:
                min_eval = float('inf')
                moves = self._get_possible_moves(-self.turn)
                for move in moves:
                    old_y, old_x, new_y, new_x = move
                    # 手を実行
                    board_state = self.board.copy_state()
                    self.board.change_cell(old_y, old_x, new_y, new_x, -self.turn)
                    
                    # 移動の評価値を計算
                    move_score = self.evaluate_move(old_y, old_x, new_y, new_x, -self.turn)
                    
                    # 再帰的に評価
                    _, eval = minimax(depth - 1, True, alpha, beta)
                    eval += move_score  # 移動の評価値を加算
                    
                    # 状態を戻す
                    self.board.restore_state(board_state)
                    
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                return best_move, min_eval
        
        # 深さ3で探索を実行
        best_move, _ = minimax(1, True, float('-inf'), float('inf'))
        return best_move

    def evaluate_move(self, old_y, old_x, new_y, new_x, current_turn):
        """手の評価値を計算"""
        score = 0
        
        # 取れる駒の評価（最も重要）
        capturable = len(self.get_capturable_pieces(new_y, new_x, current_turn))
        score += capturable * 100  # 駒を取る手を強く優先
        
        # 盤面中央への接近を評価
        center = Board.AXIS_SIZE // 2
        distance_to_center = abs(new_y - center) + abs(new_x - center)
        score += (8 - distance_to_center) * 2  # 中央に近いほど高評価
        
        # 自陣からの距離を評価
        if current_turn == Cell.UP:
            score += (Board.AXIS_SIZE - 1 - new_y) * 3  # 相手陣地に近いほど高評価
        else:
            score += new_y * 3
        
        # 端の駒を動かすことを優先
        if old_x == 0 or old_x == Board.AXIS_SIZE - 1:
            score += 5
        
        # 自分の駒が取られる危険性を評価
        if self._is_in_danger(new_y, new_x, current_turn):
            score -= 50
        
        return score

    def _is_in_danger(self, y, x, current_turn):
        """指定位置の駒が次のターンで取られる可能性があるか"""
        # 一時的に駒を配置して相手の可能な手をシミュレート
        temp_state = self.board.copy_state()
        self.board.cells[y][x].condition = current_turn
        
        # 相手の全ての可能な手をチェック
        for test_y in range(Board.AXIS_SIZE):
            for test_x in range(Board.AXIS_SIZE):
                if self.board.cells[test_y][test_x].condition == -current_turn:
                    for new_y in range(Board.AXIS_SIZE):
                        for new_x in range(Board.AXIS_SIZE):
                            if self.is_movable(test_y, test_x, new_y, new_x):
                                capturable = self.get_capturable_pieces(new_y, new_x, -current_turn)
                                if (y, x) in capturable:
                                    self.board.restore_state(temp_state)
                                    return True
        
        self.board.restore_state(temp_state)
        return False

    def _get_possible_moves(self, current_turn):
        """指定されたプレイヤーの可能な全ての手を返す"""
        moves = []
        # 盤面全体をスキャン
        for old_y in range(Board.AXIS_SIZE):
            for old_x in range(Board.AXIS_SIZE):
                # 現在のプレイヤーの駒の場合
                if self.board.cells[old_y][old_x].condition == current_turn:
                    # 移動可能な全ての位置をチェック
                    for new_y in range(Board.AXIS_SIZE):
                        for new_x in range(Board.AXIS_SIZE):
                            if self.is_movable(old_y, old_x, new_y, new_x):
                                moves.append((old_y, old_x, new_y, new_x))
        return moves

    def _evaluate_board(self):
        """現在の盤面状態を評価"""
        score = 0
        current_player = self.player1 if self.turn == self.player1.direction else self.player2
        opponent = self.player2 if self.turn == self.player1.direction else self.player1
        
        # 取った駒の数による評価
        score += (current_player.capture_count - opponent.capture_count) * 100
        
        # 3枚差が付いているかの評価
        if current_player.capture_count - opponent.capture_count >= 3:
            score += 500
        
        return score

    def _get_reward(self, captured_count):
        """報酬を計算"""
        reward = 0
        
        # 現在のプレイヤーとその相手を取得
        current_player = self.player1 if self.turn == self.player1.direction else self.player2
        opponent = self.player2 if self.turn == self.player1.direction else self.player1
        
        # 基本報酬（駒を取った場合）
        reward += captured_count * 50  # 駒を取ることの重要性を増加
        
        # 勝利状態の報酬
        winner = self.get_won_player()
        if winner is not None:
            if winner == self.turn:
                reward += 1000  # 勝利報酬を増加
            else:
                reward -= 1000  # 敗北のペナルティを増加
        
        # 3枚差の報酬
        piece_difference = current_player.capture_count - opponent.capture_count
        if piece_difference >= 3:
            reward += 300
        elif piece_difference <= -3:
            reward -= 300
        
        # 盤面の評価に基づく報酬
        board_score = self._evaluate_position()
        reward += board_score * 10
        
        return reward

    def _evaluate_position(self):
        """盤面の評価関数"""
        score = 0
        current_player = self.player1 if self.turn == self.player1.direction else self.player2
        
        # 中央支配の評価
        center = Board.AXIS_SIZE // 2
        center_range = 1
        for y in range(center - center_range, center + center_range + 1):
            for x in range(center - center_range, center + center_range + 1):
                if self.board.cells[y][x].condition == self.turn:
                    score += 2
        
        # 駒の配置の評価
        for y in range(Board.AXIS_SIZE):
            for x in range(Board.AXIS_SIZE):
                if self.board.cells[y][x].condition == self.turn:
                    # 自陣に近い駒にはペナルティ
                    if self.turn == Cell.UP:
                        score -= (Board.AXIS_SIZE - 1 - y) * 0.5
                    else:
                        score -= y * 0.5
        
        return score

    def _get_q_learning_move(self):
        """Q学習による手の選択"""
        current_player = self.player1 if self.turn == self.player1.direction else self.player2
        
        # 可能な手を取得
        possible_moves = self._get_possible_moves(self.turn)
        
        # Q学習プレイヤーから行動を取得
        return current_player.q_learning.get_action(self.board, possible_moves)

    def train_q_learning(self):
        """Q学習による学習を実行"""
        for episode in range(self.num_episodes):
            print(f"Episode {episode + 1}/{self.num_episodes}")
            
            # ボードの初期化
            self.board = Board()
            self.turn = self.player1.direction
            self.difference_3_pieces = False
            self.is_finished = False
            self.move_history = []
            
            while not self.is_finished:
                # 現在のプレイヤーのQ学習エージェントを取得
                current_player = self.player1 if self.turn == self.player1.direction else self.player2
                
                # 行動の選択
                old_y, old_x, new_y, new_x = self._get_q_learning_move()
                
                # 移動の実行
                self.board.change_cell(old_y, old_x, new_y, new_x, self.turn)
                
                # 取れる駒の処理と報酬の計算
                captured_pieces = self.get_capturable_pieces(new_y, new_x, self.turn)
                self.capture_pieces(new_y, new_x, self.turn)
                reward = self._get_reward(len(captured_pieces))
                
                # 勝利判定
                winner = self.get_won_player()
                is_done = winner is not None
                
                # Q値の更新
                current_player.q_learning.learn(self.board, reward, is_done)
                
                if is_done:
                    self.is_finished = True
                else:
                    self.change_turn()
                
                # 盤面の表示（オプション）
                if self.delay > 0:
                    import time
                    self.board.show()
                    self.show_captured_count()
                    time.sleep(self.delay)
            
            # エピソード終了時にQ-tableを保存
            self.player1.q_learning.save_q_table()
            self.player2.q_learning.save_q_table()

    def train_minimax_vs_qlearning(self):
        """MinimaxとQ学習AIの対戦による学習"""
        # 先手用と後手用のQ学習プレイヤーを別々に初期化
        q_learning_first = QLearningPlayer(Cell.UP, epsilon=0.3, alpha=0.2, gamma=0.99)
        q_learning_second = QLearningPlayer(Cell.DOWN, epsilon=0.3, alpha=0.2, gamma=0.99)
        
        # 初期の探索率を保存
        initial_epsilon = q_learning_first.epsilon
        
        for episode in range(self.num_episodes):
            # エピソード数に応じて探索率を徐々に減少
            current_epsilon = initial_epsilon * (1 - episode / self.num_episodes)
            q_learning_first.epsilon = current_epsilon
            q_learning_second.epsilon = current_epsilon
            
            print(f"Episode {episode + 1}/{self.num_episodes}")
            
            # エピソードごとに先手後手を交代
            if episode % 2 == 0:
                # Q学習が先手
                self.player1 = Player(Cell.UP, is_computer=True)
                self.player1.q_learning = q_learning_first
                self.player2 = Player(Cell.DOWN, is_computer=True)
                print("Q-Learning (UP) vs Minimax (DOWN)")
            else:
                # Q学習が後手
                self.player1 = Player(Cell.UP, is_computer=True)
                self.player2 = Player(Cell.DOWN, is_computer=True)
                self.player2.q_learning = q_learning_second
                print("Minimax (UP) vs Q-Learning (DOWN)")
            
            # ボードの初期化
            self.board = Board()
            self.turn = self.player1.direction
            self.difference_3_pieces = False
            self.is_finished = False
            self.move_history = []
            
            # 1ゲームの実行
            while not self.is_finished:
                # 現在のプレイヤーを取得
                current_player = self.player1 if self.turn == self.player1.direction else self.player2
                
                # 手の選択と実行
                old_y, old_x, new_y, new_x = self._get_computer_move()
                self.board.change_cell(old_y, old_x, new_y, new_x, self.turn)
                
                # 取れる駒の処理と報酬の計算
                captured_pieces = self.get_capturable_pieces(new_y, new_x, self.turn)
                self.capture_pieces(new_y, new_x, self.turn)
                
                # Q学習プレイヤーの場合、報酬を計算して学習
                if hasattr(current_player, 'q_learning'):
                    reward = self._get_reward(len(captured_pieces))
                    winner = self.get_won_player()
                    is_done = winner is not None
                    current_player.q_learning.learn(self.board, reward, is_done)
                
                # 勝利判定
                winner_direction = self.get_won_player()
                if winner_direction is not None:
                    self.is_finished = True
                    # 勝者のプレイヤーを特定
                    winning_player = self.player1 if winner_direction == self.player1.direction else self.player2
                    winner = "Q-Learning" if hasattr(winning_player, 'q_learning') else "Minimax"
                    print(f"Winner: {winner}")
                else:
                    self.change_turn()
                
                # 盤面の表示（オプション）
                if self.delay > 0:
                    import time
                    self.board.show()
                    self.show_captured_count()
                    time.sleep(self.delay)
            
            # エピソード終了時にQ-tableを保存
            q_learning_first.save_q_table()
            q_learning_second.save_q_table()
            
            # 結果の表示
            print(f"Final captured count - UP: {self.player1.capture_count}, DOWN: {self.player2.capture_count}")
            print("-------------------")

if __name__ == '__main__':
    play = Play()