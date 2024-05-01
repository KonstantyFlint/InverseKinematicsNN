from math import sin, cos, pi

import numpy as np
import pygame as pygame

from input_output import load
from test_generator import make_normalizer

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Arm")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Arm:

    def __init__(self, alpha, beta, length):
        self.alpha = alpha
        self.beta = beta
        self.length = length

    def draw(self):
        start_x = WIDTH / 2
        start_y = HEIGHT / 2
        mid_x = start_x + sin(self.alpha) * self.length
        mid_y = start_y + cos(self.alpha) * self.length
        end_x = mid_x - sin(self.alpha + self.beta) * self.length
        end_y = mid_y - cos(self.alpha + self.beta) * self.length

        start = (start_x, start_y)
        mid = (mid_x, mid_y)
        end = (end_x, end_y)

        pygame.draw.line(screen, BLACK, start, mid, 5)
        pygame.draw.line(screen, BLACK, mid, end, 5)


def main():
    clock = pygame.time.Clock()
    running = True

    segment_length = 150

    arm = Arm(0, 0, segment_length)
    net = load("network.pickle")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        x, y = pygame.mouse.get_pos()
        x = make_normalizer(WIDTH/2-segment_length*2, WIDTH/2 + segment_length*2, -1, 1)(x)
        y = make_normalizer(HEIGHT/2-segment_length*2, HEIGHT/2 + segment_length*2, -1, 1)(y)
        out = net.forward(np.array(np.array([x, y])))
        alpha, beta = out[0]
        normalize_angle = make_normalizer(0.1, 0.9, 0, pi)
        alpha, beta = normalize_angle(alpha), normalize_angle(beta)
        arm.alpha = alpha
        arm.beta = beta

        screen.fill(WHITE)
        arm.draw()
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
