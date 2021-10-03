import pygame as pg
import numpy, random
from enum import Enum

w = 9
h = 10
unit = 20
fx = 8
fy = 8
qmap = [[[] for y in range(w)] for x in range(h)]
map = [[0 for y in range(w)] for x in range(h)]

##every coordinate's order is y and then x

## set direction vector
class Direction(Enum):
    UP=(-1, 0)
    DOWN=(1, 0)
    LEFT=(0, -1)
    RIGHT=(0, 1)

def checkSameList(a, b):
    for n, v in a:
        if v != b[n]:
            return False
    return True

## Snake Object
class Snake:
    ## initialize
    def __init__(self) :
        self.living = True
        ## set initial position: <0, 0>
        self.position=[numpy.array([0, 3]), numpy.array([0, 2]), numpy.array([0, 1])]
        ## set initial direction
        self.dir=Direction.RIGHT.value

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
        else :
            ## check same case existing
            index = -1
            for n, c in enumerate(qmap[hy][hx]):
                if c["length"] == len(self.position) and checkSameList(c["head"], self.position[0]) and checkSameList(c["tail"], self.position[len(self.position)-1]) and checkSameList(c["apple"], [fy, fx]) and checkSameList(c["direction"], self.dir):
                    index = n
            
            if index == -1:
                self.insertCase(hy, hx)
            else :
                ## 여기에 랜덤 방향 만들어야함
                qmap[hy][hx][index]["value"]

    ## change position
    def move(self):
        self.setDirection()
        ##make next position
        next_body = [self.position[0]+self.dir] +list([x for x in self.position[0:len(self.position)-1]])
        ##make flag variable to check player living
        isLiving = self.checkCollision(next_body)

        if isLiving:
            self.deleteMyPosition()
            self.position = next_body
            self.writeMyPosition()
        else :
            self.deleteMyPosition()
            self.__init__()
            
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
            return False

        for p in self.position[1:len(self.position)-1]:
            if p[0] == nbody[0][0] and p[1] == nbody[0][1]:
                return False
        
        return True

    def changeDirection(self, type: Direction):
        self.dir=type.value
##print map
def printMap():
    print('\n'.join([''.join([str(j) for j in i]) for i in map]))


## update game data
def update(player):
    player.move()
    printMap()

## execute all drawing function
def draw(display, player):
    ##set player position into px, py 
    for part in player.position :
        py = part[0]*unit
        px = part[1]*unit
        ##px, py is left top of rectangular
        pg.draw.rect(display, (255, 255, 255), [px, py, unit, unit], 0)

    pg.draw.rect(display, (255, 255, 0), [fx*unit, fy*unit, unit, unit], 0)

if __name__ == "__main__":
    ##initialize
    pg.init()
    map[fy][fx] = 1
    snake = Snake()

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