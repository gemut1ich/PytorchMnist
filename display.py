import cv2 as cv
import numpy as np
import torch as t
import main

isMouseLBDown = False
lastPoint = (0, 0)
circleColor = (255, 255, 255)
def to_str(var):
    if type(var) is list:
        return str(var)[1:-1] # list
    if type(var) is np.ndarray:
        try:
            return str(list(var[0]))[1:-1] # numpy 1D array
        except TypeError:
            return str(list(var))[1:-1] # numpy sequence
    return str(var) # everything else


def createGrayscaleCanvas(width=500, height=500, color=0):
    canvas = np.zeros((height, width), dtype="uint8")
    return canvas


def drawCircle(event, x, y, flags, params):
    global img, isMouseLBDown, lastPoint
    if event == cv.EVENT_LBUTTONDOWN:
        isMouseLBDown = True
        cv.circle(img, (x, y), 30, (255, 255, 255), -1)
        lastPoint = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        isMouseLBDown = False
    elif event == cv.EVENT_MOUSEMOVE:
        if isMouseLBDown:
            cv.line(img, pt1=lastPoint, pt2=(x, y), color=circleColor, thickness=55)
            lastPoint = (x, y)


def predict(img2num):
    resized = cv.resize(img2num, (28, 28))
    resized = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    cv.imwrite("saveTest.png", resized)
    t_form_img = t.from_numpy(resized)
    t_form_img = t.unsqueeze(t_form_img, 0)
    t_form_img = t.unsqueeze(t_form_img, 0)
    t_form_img = t_form_img.float()

    model = main.Net()
    model.load_state_dict(t.load("mnist_cnn.pt"))
    model.eval()
    output = model(t_form_img)
    output = output.detach().numpy()
    return np.nanargmax(output)


cv.namedWindow("image")
img = np.zeros((512, 512, 3), np.uint8)
cv.setMouseCallback("image", drawCircle)
result = np.zeros((100, 100, 3), np.uint8)
cv.imshow("image", img)
cv.imshow("result", result)
while 1:
    img = np.zeros((512, 512, 3), np.uint8)
    result = np.zeros((100, 100, 3), np.uint8)
    while 1:
        cv.imshow("image", img)
        k = cv.waitKey(1)
        if k == 27:
            RESULT = predict(img)
            cv.putText(result, to_str(RESULT), (30, 65), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv.imshow("result", result)
            cv.imwrite("result.png", result)
            break



