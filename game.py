import pygame
import sys
from dataclasses import dataclass
from typing import List
import math
import random
import time

MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

SPEED = 20
RADIUS = 8
TILESIZE = 20
WIDTH, HEIGHT = TILESIZE * len(MAP[0]), TILESIZE * len(MAP)
pcsize = 10

dx = [-1,1,0,0]
dy = [0,0,1,-1]

@dataclass
class Position:
    x:int
    y:int
 
class World:
    def __init__(self, MAP:List[List]):
        self.array = MAP
        self.grid = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for y, row in enumerate(self.array):
            for x, cell in enumerate(row):
                if cell:
                    rect = pygame.draw.rect(self.grid, 'blue', (x*TILESIZE, y*TILESIZE, TILESIZE, TILESIZE))

    def draw(self, display):
        display.blit(self.grid, (0, 0))

    def check_collision(self, pacman):
        p_x, p_y = (pacman.x + pcsize)//TILESIZE, (pacman.y + pcsize)//TILESIZE
        if 0 <= p_x < len(self.array[0]) and 0 <= p_y < len(self.array):
            if self.array[p_y][p_x] == 0:
                return True
            else:
                return False
        return True

class Pellet:
    def __init__(self, x, y, radius):
        self.x, self.y = x, y
        self.radius = radius
    def draw(self, display):
        pygame.draw.circle(display, "red", (self.x, self.y), self.radius)
    def check_collision(self, obj):
        x, y = obj.x + pcsize, obj.y + pcsize
        dis = math.sqrt((self.x - x)**2 + (self.y - y)**2)
        if dis <= 2 * self.radius:
            return True
        else:
            return False


class Pacman:
    def __init__(self, pos:Position, World:World, pellets:List[Pellet]):
        self.x, self.y = pos.x, pos.y
        img = pygame.image.load("pcman.png").convert_alpha()
        self.img = pygame.transform.scale(img, (pcsize*2, pcsize*2))
        self.map = World
        self.pellets = pellets
        self.score = 0
        self.direction = "DOWN"

    def draw(self, display):
        display.blit(self.img, (self.x, self.y))

    def move(self, key:str):
        if key != "None":
            self.direction = key
        d_x, d_y = 0, 0
        if self.direction == "UP":
            d_y -= SPEED
        if self.direction == "DOWN":
            d_y += SPEED
        if self.direction == "LEFT":
            d_x -= SPEED
        if self.direction == "RIGHT":
            d_x += SPEED
        self.x += d_x
        self.y += d_y
        if self.map.check_collision(self):
            self.x -= d_x
            self.y -= d_y
        for p in self.pellets:
            if p.check_collision(self):
                self.pellets.remove(p)
                self.score += 1

class Agent:
    def __init__(self, pos:Position, World:World, pacman:Pacman):
        self.x, self.y = pos.x, pos.y
        img = pygame.image.load("pcman.png").convert_alpha()
        self.img = pygame.transform.scale(img, (pcsize*2, pcsize*2))
        self.map = World
        self.pacman = pacman
        self.direction = 0

    def draw(self, display):
        display.blit(self.img, (self.x, self.y))
    
    def check_collision(self, pacman):
        p_x, p_y = pacman.x, pacman.y
        dist = (p_x - self.x)**2 + (p_y - self.y)**2
        if dist <= 2 * pcsize:
            return True
        return False

    def move(self):
        n_x, n_y = self.x + dx[self.direction] * SPEED, self.y + dy[self.direction] * SPEED
        self.x, self.y = n_x, n_y
        if self.map.check_collision(self):
            self.x -= dx[self.direction] * SPEED
            self.y -= dy[self.direction] * SPEED

            p_x, p_y = self.pacman.x, self.pacman.y
            dist = []
            for x,y in zip(dx,dy):
                n_x = self.x + x * SPEED
                n_y = self.y + y * SPEED
                distance = (n_x - p_x)**2 + (n_y - p_y)**2
                dist.append(distance)
            dirs = [0,1,2,3]
            dirs = sorted(dirs,key = lambda x : dist[x], reverse=False)
            for i in dirs:
                self.x += dx[i] * SPEED
                self.y += dy[i] * SPEED
                if not self.map.check_collision(self):
                    self.direction = (i + random.choice([0, 1]) * random.choice([0, 1])) % 4
                    break
                else:
                    self.x -= dx[i] * SPEED
                    self.y -= dy[i] * SPEED

class Game:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('Comic Sans MS', 20)
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.map = World(MAP)
        self.reset()

    def draw(self):
        pygame.display.set_caption(f"fps:{self.clock.get_fps():.2f}")
        self.display.fill((0,0,0))
        self.map.draw(self.display)
        self.pacman.draw(self.display)
        self.agent.draw(self.display)
        for p in self.pellets:
            p.draw(self.display)
        text_surface = self.font.render(f"Score:{self.pacman.score}", False, (255, 255, 0))
        self.display.blit(text_surface, (0, 0))
        pygame.display.update()
        self.clock.tick(60)

    def draw_end(self):
        pygame.display.set_caption(f"fps:{self.clock.get_fps():.2f}")
        self.display.fill((0,0,0))
        text_surface = self.font.render(f"You fucking loser", False, (255, 255, 0))
        self.display.blit(text_surface, (0, 0))
        pygame.display.update()
        self.clock.tick(60)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if self.agent.check_collision(self.pacman):
                self.draw_end()
                time.sleep(1)
                pygame.quit()
                sys.exit()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.pacman.move("UP")
            if keys[pygame.K_DOWN]:
                self.pacman.move("DOWN")
            if keys[pygame.K_LEFT]:
                self.pacman.move("LEFT")
            if keys[pygame.K_RIGHT]:
                self.pacman.move("RIGHT")
            else:
                self.pacman.move("None")
            self.agent.move()
            self.draw()
            self.clock.tick(60)  
    
    def reset(self):
        self.pellets = [Pellet(j*TILESIZE + TILESIZE//2, i*TILESIZE + TILESIZE//2, 5) for i,r in enumerate(MAP) for j,c in enumerate(r) if c==1]
        self.pacman = Pacman(Position(4 * TILESIZE, 0), self.map, self.pellets)
        self.agent = Agent(Position(7 * TILESIZE,8 * TILESIZE), self.map, self.pacman)
        self.frame_iter = 0

    def get_state(self):
        img_arr = pygame.surfarray.array3d(pygame.display.get_surface())
        return img_arr

    def run_agent(self, move):
        done = 0
        reward = 0
        self.frame_iter += 1
        if self.agent.check_collision(self.pacman) or self.frame_iter > 300:
                done = 1
                reward = self.pacman.score
                return reward, done
        
        self.pacman.move(move)
        self.agent.move()
        self.draw()
        self.clock.tick(60)
        return self.pacman.score, done

# game = Game()
# game.run()
