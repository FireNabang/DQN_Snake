import numpy as np
from random import randrange, uniform, choice, shuffle
from enum import Enum
from DQN.dqn_agent import DQNAgent

w = 10
h = 10
unit = 50


## set direction vector
class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

Directions = [
    Direction.UP.value,
    Direction.DOWN.value,
    Direction.LEFT.value,
    Direction.DOWN.value
]


## Snake Object
class Snake:
    ## initialize
    def __init__(self):
        self.initMap()
        self.living = True
        self.position = [np.array([0, 3]), np.array([0, 2]), np.array([0, 1])]
        self.dir = Direction.RIGHT.value
        self.dir_idx = 1

        self.setFruitPosition(self.position, self.dir)
        self.epsilon = 0.99
        self.epsilon_discount = 0.95
        self.agent = DQNAgent(field_size=(h+1,w+1),batch_size=32,learning_rate=0.9,discount_factor=0.8)

    def initMap(self):
        self.map = [[-1 for y in range(w + 1)] for x in range(h + 1)]
        for i in range(1, h):
            for j in range(1, w):
                self.map[i][j] = 0

    def setFruitPosition(self, positions, dir):
        yable = list(range(1, h))
        xable = list(range(1, w))
        while True:
            shuffle(yable)
            shuffle(xable)
            new_fy, new_fx = yable[0], xable[0]
            flag = False
            for p in positions:
                if p[0] == new_fy and p[1] == new_fx:
                    flag = True
            if flag:
                continue
            else:
                self.map[new_fy][new_fx] = 1
                break

    def re(self):
        self.initMap()
        self.living = True
        self.position = [np.array([1, 3]), np.array([1, 2]), np.array([1, 1])]
        self.dir = Direction.RIGHT.value
        self.dir_idx = 1

        self.setFruitPosition(self.position, self.dir)

    def setDirection(self):
        possible = self.agent.get_q_values(self.map)
        if uniform(0, 1) < self.epsilon:
            self.dir_idx = choice(list(range(4)))
        else:
            self.dir_idx = np.argmax(possible)
        self.dir = Directions[self.dir_idx]

    ## change position
    def move(self):
        self.setDirection()

        next_body = [self.position[0] + self.dir] + list([x for x in self.position[0:len(self.position) - 1]])
        now_tail = self.position[len(self.position) - 1]
        isLiving = self.checkCollision(next_body[0])

        cur_state = np.array(self.map)
        action = self.dir_idx
        # 살아 있을 때
        if isLiving == 1:
            self.deleteMyPosition()
            self.position = next_body
            self.writeMyPosition()
            next_state = np.array(self.map)
            live = 1
            reward = 0

        # 사과 먹었을 때
        elif isLiving == 2:
            self.deleteMyPosition()
            self.position = next_body + [np.array(now_tail)]
            self.setFruitPosition(self.position, self.dir)
            self.writeMyPosition()
            next_state = self.map
            reward = len(self.position)
            live = 1

        # 죽음
        else:
            self.deleteMyPosition()
            self.position = next_body + [np.array(now_tail)]
            self.writeMyPosition()
            next_state = self.map
            reward = -1
            self.epsilon *= self.epsilon_discount
            self.re()
            live = 0
        self.agent.update_game_data(cur_state,action,reward,next_state,live)

    def writeMyPosition(self):
        for n, p in enumerate(self.position):
            if n == 0:
                dirValue = {Direction.UP.value: 3, Direction.RIGHT.value: 4, Direction.DOWN.value: 5,
                            Direction.LEFT.value: 6}
                self.map[p[0]][p[1]] = dirValue[self.dir]
            else:
                self.map[p[0]][p[1]] = 2

    def deleteMyPosition(self):
        for n, p in enumerate(self.position):
            self.map[p[0]][p[1]] = 0

    def checkCollision(self, head):

        # 탈주
        if self.map[head[0]][head[1]] == -1:
            return 0

        # 몸 충돌
        for p in self.position[1:len(self.position) - 1]:
            if p[0] == head[0] and p[1] == head[1]:
                return 0
        # 사과
        if self.map[head[0]][head[1]] == 1:
            return 2
        return 1


## update game data
def update(player):
    player.move()

if __name__ == "__main__":
    snake = Snake()
    freq = 1000
    f = 0
    while snake.agent.train_counter < 10:
        f+=1
        snake.move()
        update(snake)
        if f == freq:
            snake.agent.train()

    ## clear memory
    snake.agent.model.save_model('test_model')