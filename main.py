import pygame as pg
import numpy
from enum import Enum


## set direction vector
class Direction(Enum):
    UP=(0, -1)
    DOWN=(0, 1)
    LEFT=(-1, 0)
    RIGHT=(1, 0)

## Snake Object
class Snake:
    ## initialize
    def __init__(self) :
        ## set initial position: <0, 0>
        self.position=numpy.array([0, 0])
        ## set initial direction
        self.dir=Direction.RIGHT.value
        print(self.position+list(self.dir))
    ## change position
    def move(self):
        ## v is velocity
        v = 15
        ## change position by adding unit vector * velocity:30
        self.position+=numpy.array(self.dir)*v
    def changeDirection(self, type: Direction):
        print(type)
        self.dir=type.value

## update game data
def update(player):
    player.move()

## execute all drawing function
def draw(display, player):
    ##set player position into px, py 
    px = player.position[0]
    py = player.position[1]
    print(px, py)
    ##px, py is left top of rectangular
    pg.draw.rect(display, (255, 0, 0), [px, py, 30, 30], 0)

if __name__ == "__main__":
    ##initialize
    pg.init()
    snake = Snake()

    ##set display width and height
    display = pg.display.set_mode([400, 300])
    #set display background color

    ##set game title
    pg.display.set_caption("ex1")
    clock = pg.time.Clock()

    ## playing state flag variable
    ing = True
    
    while ing:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                ing = False
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
        clock.tick(10)

        ## update function should be followed draw function
        update(snake)
        draw(display, snake)

        ## clear display
        pg.display.flip()
        display.fill((255, 255, 255))

    ## clear memory
    pg.quit()