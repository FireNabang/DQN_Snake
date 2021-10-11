import pygame as pg
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



Directions = [
   (-1, 0),
   (1, 0),
   (0, -1),
   (0, 1)
]

## set direction vector
class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


def checkSameList(a, b):
    return a[0] == b[0] and a[1] == b[1]

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
    def __init__(self, playType):
        self.living = True
        ## set initial position: <0, 0>
        self.position = [np.array([0, 3]), np.array([0, 2]), np.array([0, 1])]
        ## set initial direction
        self.dir = Direction.RIGHT.value
        self.playType = playType

        self.epsilon = 0.99
        self.epsilon_discount = 0.95
        self.agent = DQNAgent(field_size=(h,w),batch_size=32,learning_rate=0.9,discount_factor=0.8)

    def re(self):
        self.living = True
        self.position = [np.array([0, 3]), np.array([0, 2]), np.array([0, 1])]
        self.dir = Direction.RIGHT.value

    def setDirection(self):
        possible = self.agent.get_q_values(map)
        if uniform(0, 1) < epsilon:
            self.dir = choice(possible)
        else:
            self.dir = possible.index(max(possible))


    def updateQ_Value(self):
        pass
    def chooseQvalue(self):
        pass

    ## change position
    def move(self):
        global epsilon
        if self.playType == 2:
            self.setDirection()
        ##make next position

        next_body = [self.position[0] + self.dir] + list([x for x in self.position[0:len(self.position) - 1]])
        now_tail = self.position[len(self.position) - 1]
        isLiving = self.checkCollision(next_body)

        cur_state = map
        action = Directions.index(self.dir)
        # 살아 있을 때
        if isLiving == 1:
            self.deleteMyPosition()
            self.position = next_body
            self.writeMyPosition()
            next_state = map
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
            if self.playType == 2:
                hy = self.position[0][0]
                hx = self.position[0][1]
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

    def changeDirection(self, direction: Direction):
        if self.playType == 1:
            self.dir = direction.value


##print map
def printMap():
    print('\n'.join([''.join([str(j) for j in i]) for i in map]))


## update game data
def update(player):
    player.move()
    printMap()


## execute all drawing function
def draw(display, player):
    map[fy][fx] = 1
    ##set player position into px, py
    for part in player.position:
        py = part[0] * unit
        px = part[1] * unit
        ##px, py is left top of rectangular
        pg.draw.rect(display, (255, 255, 255), [px, py, unit, unit], 0)

    pg.draw.rect(display, (255, 255, 0), [fx * unit, fy * unit, unit, unit], 0)


if __name__ == "__main__":
    choice = 1#int(input("select playing type: 1. manual 2. auto"))
    print(choice)
    ##initialize
    pg.init()
    map[fy][fx] = 1
    snake = Snake(choice)

    time = 0
    TIME = 7

    game_data = []

    ##set display width and height
    display = pg.display.set_mode([w * unit, h * unit])
    # set display background color

    ##set game title
    pg.display.set_caption("ex1")
    clock = pg.time.Clock()

    while snake.living:
        time += 1
        if time > TIME:
            time = 0
            game_data += [map]
        for event in pg.event.get():
            if event.type == pg.QUIT:
                snake.living = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    if snake.dir == Direction.RIGHT.value:
                        pass
                    else :
                        snake.changeDirection(Direction.LEFT)
                elif event.key == pg.K_RIGHT:
                    if snake.dir == Direction.LEFT.value:
                        pass
                    else:
                        snake.changeDirection(Direction.RIGHT)
                elif event.key == pg.K_UP:
                    if snake.dir == Direction.DOWN.value:
                        pass
                    else:
                        snake.changeDirection(Direction.UP)
                elif event.key == pg.K_DOWN:
                    if snake.dir == Direction.UP.value:
                        pass
                    else:
                        snake.changeDirection(Direction.DOWN)

        ##set fps
        clock.tick(8)

        ## update function should be followed draw function
        update(snake)
        draw(display, snake)

        ## clear display
        pg.display.flip()
        display.fill((0, 0, 0))

    ## clear memory
    numpy.save('./data', numpy.array(game_data))  # x_save.npy

    pg.quit()