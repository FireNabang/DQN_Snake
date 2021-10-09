import numpy as np
from random import uniform, choice, shuffle
from enum import Enum
from DQN.dqn_agent import DQNAgent
import pygame as pg

w = 10
h = 10
unit = 50


## set direction vector
class Direction(Enum):
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)

Directions = [
    Direction.UP.value,
    Direction.RIGHT.value,
    Direction.DOWN.value,
    Direction.LEFT.value
]


## Snake Object
class Snake:
    ## initialize
    def __init__(self):
        self.initMap()
        self.living = True
        self.position = [np.array([2, 3])]
        self.dir = Direction.RIGHT.value
        self.dir_idx = 1

        self.setFruitPosition(self.position)
        self.epsilon = 0.99
        self.epsilon_discount = 0.7
        self.agent = DQNAgent(field_size=(h + 2,w + 2), batch_size=32,learning_rate=0.9,discount_factor=0.8,epochs=2)


    def printMap(self):
        print('\n'.join([''.join([str(j) for j in i]) for i in self.map]))

    def initMap(self):
        self.map = [[-1 for y in range(w + 2)] for x in range(h + 2)]
        for i in range(1, h+1):
            for j in range(1, w+1):
                self.map[i][j] = 0

    def setFruitPosition(self, positions):
        yable = list(range(1, h+1))
        xable = list(range(1, w+1))
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
        self.position = [np.array([2, 3])]
        self.dir = Direction.RIGHT.value
        self.dir_idx = 1

        self.setFruitPosition(self.position)

    def setDirection(self):
        possible = self.agent.get_q_values(np.array(self.map))[0]
        if uniform(0, 1) < self.epsilon:
            temp = choice(list(range(4)))
        else:
            temp = np.argmax(possible)
        if self.dir_idx == (temp + 2) % 4:
            pass
        else:
            self.dir_idx = temp
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
            self.setFruitPosition(self.position)
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


## execute all drawing function
def draw(display, player : Snake):
    for i in range(1, h+1):
        for j in range(1, w+1):
            py = (i - 1) * unit
            px = (j - 1) * unit
            if player.map[i][j] >= 2:
                pg.draw.rect(display, (255, 255, 255), [px, py, unit, unit], 0)
            elif player.map[i][j] == 1:
                pg.draw.rect(display, (255, 255, 0), [px, py, unit, unit], 0)


if __name__ == "__main__":
    snake = Snake()
    freq = 1000
    f = 0
    display = pg.display.set_mode([w * unit, h * unit])
    pg.display.set_caption("DQN Snake!")
    clock = pg.time.Clock()
    Flag = True
    while Flag:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                Flag = False

        clock.tick(40)
        f+=1
        snake.move()
        draw(display, snake)
        ## clear display
        pg.display.flip()
        display.fill((0, 0, 0))

        if f == freq:
            snake.agent.train()
            snake.epsilon *= snake.epsilon_discount
            f = 0

    ## clear memory
    snake.agent.model.save_model('test_model')
    print('done')