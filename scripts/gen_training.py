#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from random import random, shuffle

R = 7
X_CENTER = 10
Y_CENTER = 10

X_LIMIT = 20
Y_LIMIT = 20
 
 
def inside_circle(x,y):
    return (x - X_CENTER)**2 + (y - Y_CENTER)**2 < R**2


def main():
    fp = open(sys.argv[1], 'w')      # File pointer 
    num_points = int(sys.argv[2])    # Number of points
    points = []
    
    # Setting points outside of circle
    outside = 0
    while (outside < num_points / 2):
        x = random() * X_LIMIT 
        y = random() * Y_LIMIT 
        if not inside_circle(x,y):
            points.append(((x,y), 1))
            outside += 1

    # Setting points inside of circle
    inside = 0
    while (inside < num_points / 2):
        x = random() * X_LIMIT 
        y = random() * Y_LIMIT 
        if not inside_circle(x,y):
            points.append(((x,y), -1))
            inside += 1


    
    shuffle(points)
    for ((x,y),t) in points:
        fp.write("%.6f %.6f %d\n" % (x, y, t))

    
if __name__ == "__main__":
    main()
