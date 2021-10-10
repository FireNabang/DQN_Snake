import numpy as np
from random import uniform, choice, shuffle
from enum import Enum
from DQN.dqn_agent import DQNAgent
import pygame as pg

w = 5
h = 5
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
        self.dir = [Direction.RIGHT.value]
        self.dir_idx = 1

        self.setFruitPosition(self.position)
        self.epsilon = 0.99
        self.epsilon_discount = 0.5
        self.agent = DQNAgent(field_size=(h + 2,w + 2), batch_size=32,learning_rate=0.9,discount_factor=0.8,epochs=5,data_min_size=2048)
        self.Q_value = None

    def printMap(self):
        print('\n'.join([''.join([str(j) for j in i]) for i in self.map]))

    def initMap(self):
        self.map = [[-1 for y in range(w + 2)] for x in range(h + 2)]
        for i in range(1, h+1):
            for j in range(1, w+1):
                self.map[i][j] = 1

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
                self.map[new_fy][new_fx] = 10
                break

    def re(self):
        self.initMap()
        self.living = True
        self.position = [np.array([2, 3])]
        self.dir = [Direction.RIGHT.value]
        self.dir_idx = 1

        self.setFruitPosition(self.position)

    def setDirection(self):
        possible = self.agent.get_q_values(np.array(self.map))[0]
        self.Q_value = [q[0] for q in possible]
        if uniform(0, 1) < self.epsilon:
            temp = choice(list(range(4)))
        else:
            temp = self.Q_value.index(max(self.Q_value))
        if self.dir_idx == (temp + 2) % 4:
            return Directions[self.dir_idx]
        else:
            self.dir_idx = temp
            return Directions[self.dir_idx]

    ## change position
    def move(self):
        n_dir = self.setDirection()

        next_body = [self.position[0] + self.dir[0]] + list([x for x in self.position[:-1]])
        next_dir =  [n_dir] + list([x for x in self.dir[:-1]])
        now_tail = self.position[len(self.position) - 1]
        isLiving = self.checkCollision(next_body[0])

        cur_state = np.array(self.map)
        action = self.dir_idx
        # 살아 있을 때
        if isLiving == 1:
            self.deleteMyPosition()
            self.position = next_body
            self.dir = next_dir
            self.writeMyPosition()
            next_state = np.array(self.map)
            live = 1
            reward = 0

        # 사과 먹었을 때
        elif isLiving == 2:
            self.deleteMyPosition()
            self.position = next_body + [np.array(now_tail)]
            self.dir = next_dir + [self.dir[0]]
            self.setFruitPosition(self.position)
            self.writeMyPosition()
            next_state = self.map
            reward = len(self.position)
            live = 1

        # 죽음
        else:
            self.deleteMyPosition()
            self.position = next_body + [np.array(now_tail)]
            self.dir = next_dir + [self.dir[0]]
            self.writeMyPosition()
            next_state = self.map
            reward = -1
            self.re()
            live = 0
        self.agent.update_game_data(cur_state,action,reward,next_state,live)

    def writeMyPosition(self):
        for n, p in enumerate(self.position):
            dirValue = {Direction.UP.value: 3, Direction.RIGHT.value: 5, Direction.DOWN.value: 7,
                        Direction.LEFT.value: 9}

            self.map[p[0]][p[1]] = dirValue[self.dir[n]]

    def deleteMyPosition(self):
        for n, p in enumerate(self.position):
            self.map[p[0]][p[1]] = 1

    def checkCollision(self, head):

        # 탈주
        if self.map[head[0]][head[1]] == -1:
            return 0

        # 몸 충돌
        for p in self.position[1:len(self.position) - 1]:
            if p[0] == head[0] and p[1] == head[1]:
                return 0
        # 사과
        if self.map[head[0]][head[1]] == 10:
            return 2
        return 1


## execute all drawing function
def draw(display, player : Snake):
    print(player.Q_value)
    WHITE = (255, 255, 255)
    font = pg.font.SysFont("arial", 50, True, False)
    width = (w+2) * unit
    height = (h+2) * unit
    for idx,value in enumerate(player.Q_value):
        text = font.render(str(round(value, 2)), True, WHITE)
        text_rect = text.get_rect()
        if idx == 0:
            text_rect.centerx = width + width / 2
            text_rect.top = 0
        if idx == 1:
            text_rect.right = 2 * width
            text_rect.centery = height / 2
        if idx == 2:
            text_rect.centerx = width + width / 2
            text_rect.bottom = height
        if idx == 3:
            text_rect.left = width
            text_rect.centery = height / 2
        display.blit(text, text_rect)
    for i in range(h+2):
        for j in range(w+2):
            py = i * unit
            px = j * unit
            if player.map[i][j] == -1:
                pg.draw.rect(display, (255, 0, 0), [px, py, unit, unit], 0)
            elif player.map[i][j] == 10:
                pg.draw.rect(display, (255, 255, 0), [px, py, unit, unit], 0)
            elif player.map[i][j] >= 3:
                pg.draw.rect(display, (25 * player.map[i][j], 25 * player.map[i][j], 25 * player.map[i][j]), [px, py, unit, unit], 0)
    pg.draw.rect(display, (0, 255, 0), [player.position[0][1] * unit,player.position[0][0] * unit , unit, unit], 0)


if __name__ == "__main__":
    pg.init()
    snake = Snake()
    freq = 3000
    f = 0
    display = pg.display.set_mode([(w+2 +w +2 )  * unit, (h + 2) * unit])
    pg.display.set_caption("DQN Snake!")
    clock = pg.time.Clock()
    Flag = True
    FES = 120
    while Flag:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                Flag = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    FES -= 10
                    FES = max(10, FES)
                elif event.key == pg.K_RIGHT:
                    FES += 10
                    FES = min(150, FES)
                # elif event.key == pg.K_UP:
                #     snake.changeDirection(Direction.UP)
                # elif event.key == pg.K_DOWN:
                #     snake.changeDirection(Direction.DOWN)
        clock.tick(FES)
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
    snake.agent.model.save_model('test_model2')
    print('done')