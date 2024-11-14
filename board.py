from cell import Cell

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
