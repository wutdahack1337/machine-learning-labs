class TreasureEnv():
    def __init__(self, size: int = 6):
        self.__size = size

agent = TreasureEnv(size = 10)
print(agent.__size)