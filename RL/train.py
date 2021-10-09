import numpy as np
from random import randrange, uniform, choice, shuffle
from enum import Enum
from DQN.dqn_agent import DQNAgent

w = 10
h = 10
unit = 50
fx = 5
fy = 5
map = [[0 for y in range(w)] for x in range(h)]





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


def setFruitPosition(positions, dir):
    global fy, fx
    yable = list(range(0, h))
    xable = list(range(0, w))
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
            dirValue = {Direction.UP.value: 3, Direction.RIGHT.value: 4, Direction.DOWN.value: 5,
                        Direction.LEFT.value: 6}
            map[fy][fx] = dirValue[dir]
            fy, fx = new_fy, new_fx
            map[fy][fx] = 1
            break


## Snake Object
class Snake:
    ## initialize
    def __init__(self):
        self.living = True
        self.position = [np.array([0, 3]), np.array([0, 2]), np.array([0, 1])]
        self.dir = Direction.RIGHT.value
        self.dir_idx = 1

        self.epsilon = 0.99
        self.epsilon_discount = 0.95
        self.agent = DQNAgent(field_size=(h,w),batch_size=32,learning_rate=0.9,discount_factor=0.8)

    def re(self):
        self.living = True
        self.position = [np.array([0, 3]), np.array([0, 2]), np.array([0, 1])]
        self.dir = Direction.RIGHT.value

    def setDirection(self):
        possible = self.agent.get_q_values(map)
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
        isLiving = self.checkCollision(next_body)

        cur_state = np.array(map)
        action = self.dir_idx
        # 살아 있을 때
        if isLiving == 1:
            self.deleteMyPosition()
            self.position = next_body
            self.writeMyPosition()
            next_state = np.array(map)
            live = 1
            reward = 0

        # 사과 먹었을 때
        elif isLiving == 2:
            self.deleteMyPosition()
            self.position = next_body + [np.array(now_tail)]
            setFruitPosition(self.position, self.dir)
            self.writeMyPosition()
            next_state = map
            reward = len(self.position)
            live = 1

        # 죽음
        else:
            self.deleteMyPosition()
            self.position = next_body + [np.array(now_tail)]
            setFruitPosition(self.position, self.dir)
            self.writeMyPosition()
            next_state = map
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
                print(p[0],p[1],dirValue[self.dir])
                map[p[0]][p[1]] = dirValue[self.dir]
            else:
                map[p[0]][p[1]] = 2

    def deleteMyPosition(self):
        for n, p in enumerate(self.position):
            map[p[0]][p[1]] = 0

    def checkCollision(self, nbody):
        if nbody[0][0] < 0 or nbody[0][1] < 0 or nbody[0][0] >= h or nbody[0][1] >= w:
            return 0

        for p in self.position[1:len(self.position) - 1]:
            if p[0] == nbody[0][0] and p[1] == nbody[0][1]:
                return 0

        if nbody[0][0] == fy and nbody[0][1] == fx:
            return 2
        return 1


## update game data
def update(player):
    player.move()

if __name__ == "__main__":
    map[fy][fx] = 1
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