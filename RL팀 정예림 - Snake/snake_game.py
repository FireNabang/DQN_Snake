import pygame
from pygame.locals import *
from random import randint
import os, sys

ARRAY_SIZE = 50

DIRECTIONS = { # 방향
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP": (0, 1),
    "DOWN": (0, -1),
}

snake, fruit = None, None

def init(): # 초기화
    global snake
    snake = [ (0, 2), (0, 1), (0, 0) ] # 길이가 3짜리인 snake

    place_fruit((ARRAY_SIZE // 2, ARRAY_SIZE // 2)) # 중간 지점에 fruit 배치하기

def place_fruit(coord=None): # fruit 위치를 결정하는 함수
    global fruit
    if coord: # 만약 파라미터로 fruit의 좌표가 주어진 경우라면
        fruit = coord # 그 좌표를 fruit의 위치로 설정
        return

    while True: # 위에서 return 되지 않았다는 건 fruit 좌표가 파라미터로 주어지지 않았다는 것
        x = randint(0, ARRAY_SIZE-1) # fruit의 x 좌표를 임의로 선택
        y = randint(0, ARRAY_SIZE-1) # fruit의 y 좌표를 임의로 선택
        # 근데 snake 몸이랑 겹치는 곳에 fruit을 놓으면 안되니까 while문 돌면서 안 겹치는 좌표 찾기
        if (x, y) not in snake: # snake랑 안 겹치는 fruit 좌표 찾으면
           fruit = x, y # 그 좌표를 fruit의 위치로 설정
           return

def step(direction):
    old_head = snake[0] # 기존 뱀의 머리. 예) 위의 snake = [ (0, 2), (0, 1), (0, 0) ] 라면 snake[0] = (0, 2)
    movement = DIRECTIONS[direction] # 상하좌우 중 하나. 예) (0 , 1)
    new_head = (old_head[0]+movement[0], old_head[1]+movement[1]) # 기존 뱀의 머리 + 방향 = 새로운 뱀의 머리
                                                                  # 예) (0 + 0, 2 + 1) = (0, 3) : UP!

    print(snake)
    # game over : 이동한 new 뱀의 머리가 화면 밖으로 나가거나, 자기 자신의 몸에 닿은 경우
    if (
            new_head[0] < 0 or
            new_head[0] >= ARRAY_SIZE or
            new_head[1] < 0 or
            new_head[1] >= ARRAY_SIZE or
            new_head in snake
        ):
        return False
        
    if new_head == fruit: # 이동한 new 뱀의 머리가 fruit에 닿으면 fruit 위치 재설정
        place_fruit()
    else:
        tail = snake[-1]
        del snake[-1] # 뱀 마지막 좌표 없애고,

    snake.insert(0, new_head) # 새로운 머리 위치 추가. 예)  [ (0, 2), (0, 1), (0, 0) ] -> [ (0, 3), (0, 2), (0, 1) ]

    # 뱀의 길이가 길어지는 원리 : fruit에 안 닿으면 뱀 꼬리를 자르고 새 머리를 추가하지만, fruit에 닿으면 뱀 꼬리 안 자르고 새 머리만 추가. 따라서 길이++
    return True

DIRS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

def run(): # main
    init() # 초기화

    direction = 0 # 초기 방향 설정

    pygame.init() # pygame 초기화
    s = pygame.display.set_mode((ARRAY_SIZE * 10, ARRAY_SIZE * 10)) # pygame 화면 해상도 : (50 * 10, 50 * 10)
    pygame.display.set_caption('Snake') # 타이틀 바 텍스트 설정
    appleimage = pygame.Surface((10, 10)) # fruit : 10 픽셀 * 10픽셀짜리 surface(2D 객체)
    appleimage.fill((0, 255, 0)) # fruit 색은 green
    img = pygame.Surface((10, 10)) # snake : 10 픽셀 * 10 픽셀짜리 surface(2D 객체)
    img.fill((255, 0, 0)) # snake 색은 red
    clock = pygame.time.Clock() # 게임 루프의 주기를 결정할 pygame.time.Clock 객체 생성

    pygame.time.set_timer(1, 200)

    while True:
        e = pygame.event.wait()
        if e.type == QUIT: # 게임 종료
            pygame.quit()
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 3: # 우클릭
                direction = (direction+1) % 4 # snake 우회전
            elif e.button == 1: # 좌클릭
                direction = (direction+3) % 4 # snake 좌회전

        if not step(DIRS[direction]): # step 함수에서 game over 조건, False를 return 시 게임 종료
            pygame.quit()
            sys.exit(1)

        s.fill((255, 255, 255))	# pygame 화면 하얀색으로 채우기
        for bit in snake: # snake 그리기 : snake 각 좌표마다 빨간 네모 그리기(blit)
            s.blit(img, (bit[0] * 10, (ARRAY_SIZE - bit[1] - 1) * 10)) # blit : 특정 surface의 픽셀값들을 다른 surface로 복사하는 것
        s.blit(appleimage, (fruit[0] * 10, (ARRAY_SIZE - fruit[1] - 1) * 10)) # fruit 그리기 : fruit 좌표에 초록 네모 그리기
        pygame.display.flip() # surface에 blit한 모습 보여주기(display)

run()

# 아래는 테스트용, 안 쓰는 함수들

# def print_field():
#     os.system('clear')
#     print('=' * (ARRAY_SIZE+2))
#     for y in range(ARRAY_SIZE-1, -1, -1):
#         print('|', end='')
#         for x in range(ARRAY_SIZE):
#             out = ' '
#             if (x, y) in snake:
#                 out = 'X'
#             elif (x, y) == fruit:
#                 out = 'O'
#             print(out, end='')
#         print('|')
#     print('=' * (ARRAY_SIZE+2))

# def test():
#     global fruit
#     init()
#     assert step('UP')

#     assert snake == [(0, 3), (0, 2), (0, 1)]

#     fruit = (0, 4)
#     assert step('UP')

#     assert snake == [(0, 4), (0, 3), (0, 2), (0, 1)]
#     assert fruit != (0, 4)

#     assert not step('DOWN'), 'noo!'