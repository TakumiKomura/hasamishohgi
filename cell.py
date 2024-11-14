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
        