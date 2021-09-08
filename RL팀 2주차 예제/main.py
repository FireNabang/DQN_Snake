import sys
import pygame
from pygame.locals import KEYDOWN, K_q
from dataclasses import dataclass
from random import randrange, uniform, choice

TILE_SIZE = 100

WIDTH = TILE_SIZE*12
HEIGHT = TILE_SIZE*9
WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)



clock = pygame.time.Clock()
grid = pygame.display.set_mode((WIDTH, HEIGHT))

Dir = ['UP','RIGHT','DOWN','LEFT']

@dataclass
class Point:
    x : int = 0
    y: int = 0



def main():
    init()
    while True:
        clock.tick(20)
        drawTile() # 타일 그리기
        drawGrid() # 그리드(격자) 그리기
        checkEvents()
        movePlayer() # 이 함수를 주석 처리하면 방향키로 Agent를 움직일 수 있습니다.
        displayQ() # Q값 표시
        checkGame()
        pygame.display.update()

def init():
    pygame.init()
    global  Sx,Sy
    global env,prePlayer, player,Q_Value, epsilon,epsilon_discount,learning_rate,discount_factor
    # 여기서 파라미터들을 수정해보시길 바랍니다!
    epsilon = 0.99       
    epsilon_discount = 0.95
    learning_rate = 0.8
    discount_factor = 0.9
    Sx,Sy = 0,0 # player 시작점
    player = Point(Sx, Sy)
    prePlayer = Point()
    Q_Value = []

    for i in range(0, HEIGHT, TILE_SIZE):
        temp = []
        for j in range(0, WIDTH, TILE_SIZE):
            temp += [{'RIGHT' : 0,'LEFT' : 0,'UP' : 0,'DOWN' : 0}]
        Q_Value += [temp]
    env = []
    for i in range(0, HEIGHT, TILE_SIZE):
        temp = []
        for j in range(0, WIDTH, TILE_SIZE):
            temp += [0]
        env += [temp]
        
    while True:
        tx = randrange(0, WIDTH // TILE_SIZE)
        ty = randrange(0, HEIGHT // TILE_SIZE)
        if tx == player.x or ty == player.y:
            continue
        env[ty][tx] = -1
        break

    while True:
        tx = randrange(0, WIDTH // TILE_SIZE)
        ty = randrange(0, HEIGHT // TILE_SIZE)
        if tx == player.x or ty == player.y or env[ty][tx] == -1:
            continue
        env[ty][tx] = 1
        break
    cnt = 20
    # 랜덤으로 벽을 세워 줍니다
    while cnt > 0:
        tx = randrange(0, WIDTH // TILE_SIZE)
        ty = randrange(0, HEIGHT // TILE_SIZE)
        if tx == player.x or ty == player.y or env[ty][tx] == -1 or env[ty][tx] == 1:
            continue
        env[ty][tx] = 2
        cnt -= 1


def drawGrid():
    for x in range(0, WIDTH, TILE_SIZE):
        pygame.draw.line(grid, BLACK, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, TILE_SIZE):
        pygame.draw.line(grid, BLACK, (0, y), (WIDTH, y))

def drawTile():
    for i in range(0,HEIGHT//TILE_SIZE):
        for j in range(0, WIDTH//TILE_SIZE):
            if env[i][j] == 0:
                pygame.draw.rect(grid,WHITE,[j*TILE_SIZE,i*TILE_SIZE,TILE_SIZE,TILE_SIZE])
            elif env[i][j] == -1:
                pygame.draw.rect(grid,RED,[j*TILE_SIZE,i*TILE_SIZE,TILE_SIZE,TILE_SIZE])
            elif env[i][j] == 1:
                pygame.draw.rect(grid,GREEN,[j*TILE_SIZE,i*TILE_SIZE,TILE_SIZE,TILE_SIZE])
            elif env[i][j] == 2:
                pygame.draw.rect(grid,GRAY,[j*TILE_SIZE,i*TILE_SIZE,TILE_SIZE,TILE_SIZE])

    pygame.draw.rect(grid,BLUE,[player.x*TILE_SIZE,player.y*TILE_SIZE,TILE_SIZE,TILE_SIZE])

    
# 키보드 입력 처리
def checkEvents():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == KEYDOWN and event.key == K_q:
            pygame.quit()
            sys.exit()
    key_event = pygame.key.get_pressed()
    if key_event[pygame.K_LEFT]:
        if player.x <= 0 or env[player.y][player.x-1] == 2:
            return
        prePlayer.x, prePlayer.y = player.x, player.y
        player.x -= 1
        updateQ_Value(prePlayer.x, prePlayer.y, 'LEFT')
    if key_event[pygame.K_RIGHT]:
        if player.x >= WIDTH//TILE_SIZE -1 or env[player.y][player.x+1] == 2:
            return
        prePlayer.x, prePlayer.y = player.x, player.y
        player.x += 1
        updateQ_Value(prePlayer.x, prePlayer.y, 'RIGHT')
    if key_event[pygame.K_UP]:
        if player.y <= 0 or env[player.y-1][player.x] == 2:
            return
        prePlayer.x, prePlayer.y = player.x, player.y
        player.y -= 1
        updateQ_Value(prePlayer.x, prePlayer.y, 'UP')
    if key_event[pygame.K_DOWN]:
        if player.y >= HEIGHT//TILE_SIZE - 1 or env[player.y+1][player.x] == 2:
            return
        prePlayer.x, prePlayer.y = player.x, player.y
        player.y +=1
        updateQ_Value(prePlayer.x, prePlayer.y, 'DOWN')
    if key_event[pygame.QUIT]:
        sys.exit()
    if key_event[pygame.K_q]:
        pygame.quit()
        sys.exit()

        
def movePlayer():
    if uniform(0, 1) < epsilon:
        nDir = choice(Dir)
    else :
        max(Q_Value[player.y][player.x], key=Q_Value[player.y][player.x].get)
        nDir = choice([k for k,v in Q_Value[player.y][player.x].items() if max(Q_Value[player.y][player.x].values()) == v])

    if nDir == 'LEFT':
        if player.x <= 0 or env[player.y][player.x - 1] == 2:
            return
        prePlayer.x, prePlayer.y = player.x, player.y
        player.x -= 1
    if nDir == 'RIGHT':
        if player.x >= WIDTH // TILE_SIZE - 1 or env[player.y][player.x + 1] == 2:
            return
        prePlayer.x, prePlayer.y = player.x, player.y
        player.x += 1
    if nDir == 'UP':
        if player.y <= 0 or env[player.y - 1][player.x] == 2:
            return
        prePlayer.x, prePlayer.y = player.x, player.y
        player.y -= 1
    if nDir == 'DOWN':
        if player.y >= HEIGHT // TILE_SIZE - 1 or env[player.y + 1][player.x] == 2:
            return
        prePlayer.x, prePlayer.y = player.x, player.y
        player.y += 1

    updateQ_Value(prePlayer.x,prePlayer.y,nDir)


# 자료에 있는 수식을 그대로 코드로 옮겼습니다
def updateQ_Value(curX,curY,action):
    Q_Value[curY][curX][action] = (1-learning_rate)*Q_Value[curY][curX][action] + learning_rate*(env[player.y][player.x] + discount_factor * max(Q_Value[player.y][player.x].values())) 


# 방향에 맞게 Q값을 보여줍니다
def displayQ():
    font = pygame.font.SysFont("arial", 20, True, False)

    for i in range(0, HEIGHT // TILE_SIZE):
        for j in range(0, WIDTH // TILE_SIZE):
            for dir in Dir:
                text = font.render(str(round(Q_Value[i][j][dir], 2)), True, BLACK)
                text_rect = text.get_rect()
                if dir == 'LEFT':
                    text_rect.left = j * TILE_SIZE
                    text_rect.centery = i * TILE_SIZE + TILE_SIZE / 2
                elif dir == 'RIGHT':
                    text_rect.right = (j+1) * TILE_SIZE
                    text_rect.centery = i * TILE_SIZE + TILE_SIZE / 2
                elif dir == 'UP':
                    text_rect.centerx = j * TILE_SIZE + TILE_SIZE / 2
                    text_rect.top = i * TILE_SIZE
                elif dir == 'DOWN':
                    text_rect.centerx = j * TILE_SIZE + TILE_SIZE / 2
                    text_rect.bottom = (i + 1) * TILE_SIZE
                grid.blit(text, text_rect)


# agent가 -1 or 1에 도착했을경우 epsilon을 감소시키고 시작점으로 돌아갑니다
def checkGame():
    global epsilon
    if env[player.y][player.x] == -1 or env[player.y][player.x] == 1:
        player.x = Sx
        player.y = Sy
        epsilon *= epsilon_discount

if __name__ == '__main__':
    main()

