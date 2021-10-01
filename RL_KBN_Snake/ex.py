import pygame as pg
import numpy
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

## Snake Object
class Snake:
    ## initialize
    def __init__(self) :
        self.living = True
        ## set initial position: <0, 0>
        self.position=[numpy.array([0, 3]), numpy.array([0, 2]), numpy.array([0, 1])]
        ## set initial direction
        self.dir=Direction.RIGHT.value
    ## change position
    def move(self):        
        self.position = [self.position[0]+numpy.array(self.dir), self.position[0], self.position[1]]

        for p in self.position:
            if len(list(filter(lambda x: self.position[x][0] == p[0] and self.position[x][1] == p[1], range(len(self.position))))) > 1 :
                self.living = False
            

        if self.position[0][0] == fy and self.position[0][1] == fx :
            ##self.position.append(last)
            self.__init__()

    def changeDirection(self, type: Direction):
        self.dir=type.value

## update game data
def update(player):
    player.move()

    for n, i in enumerate(player.position[0]):
        if n==0 and (i < 0 or i > w):
            player.living = False
        if n==1 and (i < 0 or i > h):
            player.living = False

    print('\n'.join([ ''.join([str(j) for j in i]) for i in map]))

## execute all drawing function
def draw(display, player):
    ##set player position into px, py 
    for part in player.position :
        py = part[0]*unit
        px = part[1]*unit
        ##px, py is left top of rectangular
        pg.draw.rect(display, (255, 255, 255), [px, py, unit, unit], 0)

    pg.draw.rect(display, (255, 255, 0), [fx*unit, fy*unit, unit, unit], 0)
    ##for i in map:
    ##    for j in i:
    ##        if numpy.array([j, i]) in player.position:
    ##            print("x", end="")
    ##        else:
    ##            print("0", end="")
    ##    print('')

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