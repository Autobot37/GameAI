import numpy as np
import pygame
import sys
from pygame.locals import *
import math
import numpy as np
import random

TILESIZE = 16
NROWS = 36
NCOLS = 28
SCREENWIDTH = NCOLS * TILESIZE
SCREENHEIGHT = NROWS * TILESIZE
Yellow = (255, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

class Vector: 
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    def asInt(self):
        return (int(self.x), int(self.y))
    def __mul__(self, num):
        return Vector(self.x * num, self.y * num)
    def copy(self):
        return Vector(self.x, self.y)
    def squared(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

directions = {"STOP":Vector(), "UP":Vector(0, -1), "DOWN":Vector(0, 1), "LEFT":Vector(-1, 0), "RIGHT":Vector(1, 0)}

class Entity(object):
    def __init__(self, node):
        self.name = "PACMAN"
        self.node = node
        self.position = self.node.position
        self.direction = "STOP"
        self.speed = 10
        self.radius = 10
        self.color = Yellow
        self.target = node
        self.middir = "STOP"
        self.score = 0
        
    def render(self, display):
        pygame.draw.circle(display, self.color, self.position.asInt(), self.radius)
    def update(self, move=None):
        if move is not None:
            direction = move
        else:
            direction = self.getkey()
        if direction == "STOP":
            direction = self.direction
        if direction != "STOP":
            self.middir = direction

        if self.position == self.node.position:
            if self.node.neighbors[direction] is not None:
                self.direction = direction
                self.target = self.node.neighbors[direction]
                self.position += directions[direction] * self.speed
            else:
                self.direction = "STOP"
        else:
            #ifreverse then reverse it else 
            self.position += directions[self.direction] * self.speed
            if self.overshoot():
                self.direction = "STOP"
                self.node = self.target
                self.position = self.node.position

            if (directions[direction] - directions[self.direction]).squared() == 4:
                self.direction = direction
                self.node, self.target = self.target, self.node
        if self.position == self.node.position and direction =="STOP":
            self.direction = self.middir

    def overshoot(self):
        vec1 = self.target.position - self.node.position
        vec2 = self.node.position - self.position
        vec1 = vec1.squared()
        vec2 = vec2.squared()
        return vec2 >= vec1

    def getkey(self):
        key = pygame.key.get_pressed()
        if key[K_UP]:
            return "UP"
        if key[K_DOWN]:
            return "DOWN"
        if key[K_LEFT]:
            return "LEFT"
        if key[K_RIGHT]:
            return "RIGHT"
        return "STOP"

class Pacman(Entity):
    def __init__(self,node):
        Entity.__init__(self, node)

class Ghost(Entity):
    def __init__(self,node):
        Entity.__init__(self, node)
    def getkey(self):
        return random.choice(["UP","DOWN","LEFT","RIGHT"])

class Node:
    def __init__(self,x,y):
        self.position = Vector(x,y)
        self.neighbors = {"STOP":None,"UP":None,"DOWN":None,"LEFT":None,"RIGHT":None}
    def render(self, screen):
        pygame.draw.circle(screen, RED, self.position.asInt(), 5)
        for key,val in self.neighbors.items():
            if val is not None:
                pygame.draw.line(screen, WHITE, self.position.asInt(), val.position.asInt())

class NodeGroup:
    def __init__(self):
        self.nodesgrp = {}
        self.nodesymbol = ['+','n','P']
        self.pathsymbol = ['.','-','|','p']
        self.maze = np.loadtxt("maze.txt", dtype=str, delimiter=' ') 
        self.connecthori()
        self.connectvert()
    def connecthori(self):
        for i in range(len(self.maze)):
            prev = None
            for j in range(len(self.maze[0])):
                if self.maze[i][j] in self.nodesymbol:
                    if (i,j) in self.nodesgrp:
                        node = self.nodesgrp[(i,j)]
                    else:
                        node = Node(j*TILESIZE, i*TILESIZE)
                    if prev is not None and self.maze[i][j-1] in self.pathsymbol:
                        prev.neighbors["RIGHT"] = node
                        node.neighbors["LEFT"] = prev
                    self.nodesgrp[(i, j)] = node
                    prev = node

    def connectvert(self):
        for j in range(len(self.maze[0])):
            prev = None
            for i in range(len(self.maze)):
                if self.maze[i][j] in self.nodesymbol:
                    if (i,j) in self.nodesgrp:
                        node = self.nodesgrp[(i,j)]
                    else:
                        node = Node(j*TILESIZE, i*TILESIZE)
                    if prev is not None and self.maze[i-1][j] in self.pathsymbol:
                        prev.neighbors["DOWN"] = node
                        node.neighbors["UP"] = prev
                    self.nodesgrp[(i, j)] = node
                    prev = node

    def render(self, screen):
        for node in self.nodesgrp.values():
            node.render(screen)

class Pellet:
    def __init__(self, x, y):
        self.position = Vector(x,y)
        self.color = WHITE
        self.radius = 4
    def render(self, screen):
        pygame.draw.circle(screen, self.color, self.position.asInt(), self.radius)

class Pellets:
    def __init__(self):
        self.pellets = []
        self.create()
    def create(self):
        data = np.loadtxt("maze.txt", dtype=str, delimiter=" ")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i][j] in  ['.', 'p']:
                    self.pellets.append(Pellet(j*TILESIZE, i*TILESIZE))
    def render(self, screen):
        for p in self.pellets:
            p.render(screen)
    def update(self, pacman):
        position = pacman.position
        for pelet in self.pellets:
            dis = (position - pelet.position).squared()
            if dis <= 2*pelet.radius:
                self.pellets.remove(pelet)
                pacman.score += 1

   
        
class Game:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        self.clock = pygame.time.Clock()
        self.start()

    def start(self):
        self.screen.fill((0,0,0))
        self.frame_iter = 0
        self.nodes = NodeGroup()
        self.pacman = Pacman(list(self.nodes.nodesgrp.values())[0])
        self.pellets = Pellets()
        self.ghost = Ghost(list(self.nodes.nodesgrp.values())[-1])
    
    def update(self):
        while True:
            self.checkevents()
            self.render()
            self.pacman.update()
            self.ghost.update()
            self.pellets.update(self.pacman)

    
    def checkevents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def render(self):
        self.screen.fill((0,0,0))
        pygame.display.set_caption(f"Fps:{self.clock.get_fps() : .2f}")
        self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        self.pacman.render(self.screen)
        self.ghost.render(self.screen)
        pygame.display.update()
        self.clock.tick(60)

    def get_state(self):
        img_arr = pygame.surfarray.array3d(pygame.display.get_surface())
        return img_arr
    def run_agent(self, move):
        pygame.event.pump()
        done = 0
        reward = 0
        self.frame_iter += 1
        if self.frame_iter > 150:
                done = 1
                reward = -10
                return reward, self.pacman.score , done
        
        self.render()
        self.pacman.update(move)
        self.ghost.update()
        self.pellets.update(self.pacman)        
        self.clock.tick(60)
        return self.pacman.score, self.pacman.score, done

if __name__ == "__main__":
    game = Game()
    game.update()

