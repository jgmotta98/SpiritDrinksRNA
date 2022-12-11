"""
Created on Fri Nov 02 00:07:47 2022
"""
# The algorithm captures the image of one or two (optional) webcams, treats and stores
# mean RGB and HSV data in an Excel worksheet.

import cv2
import numpy as np
import pandas as pd

# Capture chosen webcam, get a frame, modify the frame size, save uncut and cut frame as jpg,
# return cut frame BGR matrix and sample name.
def camCapture(sampleName, camNumber, camWidth, camHeight):
    webcam = cv2.VideoCapture(camNumber)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)
    isWorking, frame = webcam.read()
    cv2.imshow("Webcam image", frame)
    # Press any key to close the image window visualization.
    cv2.waitKey(0)
    cv2.imwrite(str(sampleName) + '.jpg', frame)
    framecut = frame[0:140, 0:639]
    cv2.imwrite(str(sampleName) + '_cut.jpg', framecut)
    cv2.destroyAllWindows()
    return framecut, sampleName

# Receive frame BGR data, converts BGR to RGB, converts RGB to HSV 8bit,
# converts HSV 8bit to real HSV. Return mean RGB, mean HSV and sample name.
def rgbToHsv(framesData):
    frameRgbWebcam = cv2.cvtColor(framesData[0], cv2.COLOR_BGR2RGB)
    meanRgbWebcam = np.mean(frameRgbWebcam, axis=(0, 1))
    frameHsvWebcam = cv2.cvtColor(frameRgbWebcam, cv2.COLOR_RGB2HSV)
    # Destination HSV data type from cv2.COLOR_RGB2HSV is: H/2 → H; 255S → S; 255V → V.
    # Conversion is needed.
    for i, framesHsv1 in enumerate(frameHsvWebcam):
        for j, framesHsv2 in enumerate(framesHsv1):
            framesHsv2[0] = framesHsv2[0] * 2
            framesHsv2[1] = (framesHsv2[1] * 100) / 255
            framesHsv2[2] = (framesHsv2[2] * 100) / 255
    meanHsvWebcam = np.mean(frameHsvWebcam, axis=(0, 1))
    return meanRgbWebcam, meanHsvWebcam, framesData[1]

# Receive 1st cam data and 2nd cam data (optional), save data into an Excel file.
def saveToExcel(firstWebcamMeanData, secondWebcamMeanData = None):
    oldSamples = pd.read_excel("DataCV.xls")
    rgbCam1 = np.transpose(pd.DataFrame(firstWebcamMeanData[0]))
    hsvCam1 = np.transpose(pd.DataFrame(firstWebcamMeanData[1]))
    if secondWebcamMeanData:
        rgbCam2 = np.transpose(pd.DataFrame(secondWebcamMeanData[0]))
        hsvCam2 = np.transpose(pd.DataFrame(secondWebcamMeanData[1]))
    else:
        rgbCam2 = hsvCam2 = [None]*3
    finalData = pd.DataFrame({'Name': firstWebcamMeanData[2], 'valueR_cam1': rgbCam1[0],
                              'valueG_cam1': rgbCam1[1],'valueB_cam1': rgbCam1[2],
                              'valueH_cam1': hsvCam1[0],'valueS_cam1': hsvCam1[1],
                              'valueV_cam1': hsvCam1[2],'valueR_cam2': rgbCam2[0],
                              'valueG_cam2': rgbCam2[1],'valueB_cam2': rgbCam2[2],
                              'valueH_cam2': hsvCam2[0],'valueS_cam2': hsvCam2[1],
                              'valueV_cam2': hsvCam2[2]})
    gatheredData = pd.concat([oldSamples, finalData])
    gatheredData.to_excel('DataCV.xls', index=False)

# 1st Camera.
frameData1 = camCapture('AmostraTeste', 0, 720, 720)
meanDataCam1 = rgbToHsv(frameData1)

# 2nd Camera.
frameData2 = camCapture('AmostraTesteCam2', 1, 720, 720)
meanDataCam2 = rgbToHsv(frameData2)

# Saving to Excel file.
saveToExcel(meanDataCam1, meanDataCam2)