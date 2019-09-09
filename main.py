# coding:utf-8

from RandomErasingGenerator import RandomErasingGenerator
import cv2

if __name__ == "__main__":
    reg = RandomErasingGenerator("", (256, 256), 10)
    X, y = reg.__getitem__(1)
    counter = 0
    for x in X:
        cv2.imwrite("{}.png".format(counter), x)
        counter += 1
