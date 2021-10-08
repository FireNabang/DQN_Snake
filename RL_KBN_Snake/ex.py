import pygame as pg
import numpy, random
from enum import Enum

w = 9
h = 10
unit = 20
fx = 5
fy = 5
qmap = [[[] for y in range(w)] for x in range(h)]
map = [[0 for y in range(w)] for x in range(h)]
epsillon = 0.9
discount_factor = 0.95


##every coordinate's order is y and then x

## set direction vector
class Direction(Enum):
    UP=(-1, 0)
    DOWN=(1, 0)
    LEFT=(0, -1)
    RIGHT=(0, 1)

def checkSameList(a, b):
    return a[0] == b[0] and a[1] == b[1]

def setFruitPosition(positions, dir):
    global fy, fx
    yable = list(range(0, h))
    xable = list(range(0, w))
    while True:
        random.shuffle(yable)
        random.shuffle(xable)
        new_fy, new_fx = yable[0], xable[0]
        flag = False
        for p in positions:
            if p[0] == new_fy and p[1] == new_fx:
                flag = True
        if flag:
            continue
        else :
            dirValue = {Direction.UP.value: 3, Direction.RIGHT.value: 4, Direction.DOWN.value: 5, Direction.LEFT.value: 6}
            map[fy][fx] = dirValue[dir]
            fy, fx = new_fy, new_fx
            map[fy][fx] = 1
            print("끝")
            break

## Snake Object
class Snake:
    ## initialize
    def __init__(self, playType) :
        self.living = True
        ## set initial position: <0, 0>
        self.position=[numpy.array([0, 3]), numpy.array([0, 2]), numpy.array([0, 1])]
        ## set initial direction
        self.dir=Direction.RIGHT.value
        self.playType = playType

    def insertCase(self, hy, hx):
        qmap[hy][hx].append({
            "length" : len(self.position),
            "head": self.position[0],
            "tail": self.position[len(self.position)-1],
            "apple": [fy, fx],
            "direction": self.dir,
            "value": {Direction.LEFT.value: 0, Direction.RIGHT.value: 0, Direction.UP.value: 0, Direction.DOWN.value: 0}
        })


    def setDirection(self):
        ##set y of head, x of head into hy, hx
        hy = self.position[0][0]
        hx = self.position[0][1]
        ##check that qmap has any case
        ##if it doesn't have case
        if len(qmap[hy][hx])==0:
            self.insertCase(hy, hx)
        ##if it has a case or more
        ## check same case existing
        index = -1
        for n, c in enumerate(qmap[hy][hx]):
            if c["length"] == len(self.position) and checkSameList(c["head"], self.position[0]) and checkSameList(c["tail"], self.position[len(self.position)-1]) and checkSameList(c["apple"], [fy, fx]) and c["direction"] == self.dir:
                index = n
        
        if index == -1:
            self.insertCase(hy, hx)

        for k, v in qmap[hy][hx][index]["value"].items():
            if v == 1:
                self.dir = k
                return
        possible = [ k for k, v in qmap[hy][hx][index]["value"].items() if v>=0]
        inable = []
        for pn, p in enumerate(possible):
            next_head = self.position[0] + p
            print(possible, [fy, fx])
            if next_head[0] == fy and next_head[1] == fx:
                print("사과 근처", p, self.position)
                qmap[hy][hx][index]["value"][p] = 1
                self.dir = p
                return
            if self.position[1][0] == next_head[0] and self.position[1][1] == next_head[1]:
                inable.append(pn)
        for x in inable:
            qmap[hy][hx][index]["value"][possible[x]] = -1
            del possible[x]
        random.shuffle(possible)
        self.dir = possible[0]
            
            

    def updateQ(self, case):
        if case == 1:
            for n, c in enumerate(qmap[self.position[0][0]][self.position[0][1]]):
                if c["length"] == len(self.position) and checkSameList(c["head"], self.position[0]) and checkSameList(c["tail"], self.position[len(self.position)-1]) and checkSameList(c["apple"], [fy, fx]) and c["direction"] == self.dir:
                    c["value"][self.dir] = 1

    ## change position
    def move(self):
        if self.playType == 2:
            self.setDirection()
        ##make next position
        next_body = [self.position[0]+self.dir] +list([x for x in self.position[0:len(self.position)-1]])
        now_tail = self.position[len(self.position)-1]
        ##make flag variable to check player living
        isLiving = self.checkCollision(next_body)

        if isLiving == 1:
            self.deleteMyPosition()
            self.position = next_body
            self.writeMyPosition()

        elif isLiving == 2:
            self.deleteMyPosition()
            self.position = next_body + [numpy.array(now_tail)]
            if self.playType == 1:
                setFruitPosition(self.position, self.dir)
            self.writeMyPosition()

        else :
            self.deleteMyPosition()
            if self.playType == 2:
                hy = self.position[0][0]
                hx = self.position[0][1]
                for c in qmap[hy][hx]:
                    if c["length"] == len(self.position) and checkSameList(c["head"], self.position[0]) and checkSameList(c["tail"], self.position[len(self.position)-1]) and checkSameList(c["apple"], [fy, fx]) and checkSameList(c["direction"], self.dir):
                        c["value"][self.dir] = -1
            self.__init__(self.playType)
            
    def writeMyPosition(self):
        for n, p in enumerate(self.position):
            if n == 0:
                dirValue = {Direction.UP.value: 3, Direction.RIGHT.value: 4, Direction.DOWN.value: 5, Direction.LEFT.value: 6}
                map[p[0]][p[1]] = dirValue[self.dir]
            else :
                map[p[0]][p[1]] = 2

    def deleteMyPosition(self):
        for n, p in enumerate(self.position):
            map[p[0]][p[1]] = 0
    
    def checkCollision(self, nbody):
        if nbody[0][0] < 0 or nbody[0][1] < 0 or nbody[0][0] >= h or nbody[0][1] >= w :
            return 0

        for p in self.position[1:len(self.position)-1]:
            if p[0] == nbody[0][0] and p[1] == nbody[0][1]:
                return 0
        
        if nbody[0][0] == fy and nbody[0][1] == fx :
            return 2

        return 1

    def changeDirection(self, direction: Direction):
        if self.playType==1:
            self.dir=direction.value

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
    for part in player.position :
        py = part[0]*unit
        px = part[1]*unit
        ##px, py is left top of rectangular
        pg.draw.rect(display, (255, 255, 255), [px, py, unit, unit], 0)

    pg.draw.rect(display, (255, 255, 0), [fx*unit, fy*unit, unit, unit], 0)

if __name__ == "__main__":
    choice = int(input("select playing type: 1. manual 2. auto"))
    print(choice)
    ##initialize
    pg.init()
    map[fy][fx] = 1
    snake = Snake(choice)

    ##set display width and height
    display = pg.display.set_mode([w*unit, h*unit])
    #set display background color

    ##set game title
    pg.display.set_caption("ex1")
    clock = pg.time.Clock()
    
    while snake.living:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                snake.living = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    snake.changeDirection(Direction.LEFT)
                elif event.key == pg.K_RIGHT:
                    snake.changeDirection(Direction.RIGHT)
                elif event.key == pg.K_UP:
                    snake.changeDirection(Direction.UP)
                elif event.key == pg.K_DOWN:
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
    pg.quit()