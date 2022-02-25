import tensorflow as tf
import numpy as np
import cv2
import pyautogui
import time

def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True

def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None

model = tf.keras.models.load_model('models/mnist')
print("Model Loaded")

myScreenshot = pyautogui.screenshot()
myScreenshot.save('images\\temp.png')

gridImg=cv2.imread('images\\temp.png')
gridImg=gridImg[260:790,294:825]
gridImg=cv2.cvtColor(gridImg,cv2.COLOR_BGR2GRAY)
cv2.imwrite("images\\grid.png",gridImg)
print("Grid Located")

y=0
cellSize=59
grid=[]
for i in range(9):
    x=0
    row=[]
    temp=[]
    for j in range(9):
        cell = cv2.adaptiveThreshold(gridImg[y+2:y+55,x+2:x+57],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        cell=cv2.bitwise_not(cell)
        contours, hierarchy = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if np.mean(cell)<=5:
            row.append(0)
            temp.append(0)
        else:
            cell = cv2.resize(cell, (28,28), interpolation= cv2.INTER_LINEAR)
            cell = cell.reshape(28,28,1)
            pred=model.predict(np.array([cell])/255)
            if np.argmax(pred[0])==0:
                row.append(6)
            elif np.argmax(pred[0])==5:
                if pred[0,5]>0.95:
                    row.append(5)
                else:
                    row.append(6)
            elif np.argmax(pred[0])==9:
                if pred[0,9]>0.85:
                    row.append(9)
                else:
                    row.append(6)
            else:
                row.append(np.argmax(pred[0]))
        x+=59
    y+=59
    grid.append(row)
print("Grid Image Converted to List")
for i in grid:
    print(i)

solve(grid)
print("Soduku Solved")
for i in grid:
    print(i)

y=290

for i in range(9):
    x=325
    for j in range(9):
        pyautogui.click(x,y)
        time.sleep(0.05)
        pyautogui.press(str(grid[i][j]))
        time.sleep(0.05)
        x+=60
    y+=60
print("Done")