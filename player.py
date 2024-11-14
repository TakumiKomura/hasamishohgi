from cell import Cell
from board import Board

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
