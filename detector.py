import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import numpy as np
import time
import pyautogui

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("model_program/keras_model.h5", "model_program/labels.txt")

labels_dict = {}
with open("model_program/labels.txt", "r") as f:
	for line in f:
		index, label = line.strip().split()
		labels_dict[int(index)] = label

offset = 20
img_size = 300

label_list = []
counter = 0


def normalize_image(img_crop, img_white, h, w):
	if img_crop is not None and img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
		aspect_ratio = h/w 
		#This just makes the images centered and constrained to image_size box (300 x 300)
		if aspect_ratio > 1:
			k = img_size/h
			w_cal = math.ceil(k*w)
			if w_cal > 0:
				img_resize = cv2.resize(img_crop, (w_cal, img_size))
				img_resize_shape = img_resize.shape
				w_gap = math.ceil((img_size - w_cal) / 2)
				img_white[:, w_gap:w_cal + w_gap] = img_resize
		else:
			k = img_size/w
			h_cal = math.ceil(k*h)
			if h_cal > 0:
				img_resize = cv2.resize(img_crop, (img_size, h_cal))
				img_resize_shape = img_resize.shape
				h_gap = math.ceil((img_size - h_cal) / 2)
				img_white[h_gap:h_cal + h_gap, :] = img_resize
	return img_white

while True:
	success, img = cap.read()
	hands, img = detector.findHands(img)
	
	if hands:
		hand = hands[0]
		x,y,w,h = hand['bbox']
		img_crop = None
		img_white = np.ones((img_size, img_size, 3), np.uint8)*255
		
		if len(hands) == 2:
			hand2 = hands[1]
			x2,y2,w2,h2 = hand2['bbox']
			
			x_min = max(0, min(x, x2) - offset)
			y_min = max(0, min(y, y2) - offset)
			x_max = max(img.shape[1], max(x+w, x2+w2) + offset)
			y_max = max(img.shape[0], max(y+h, y2+h2) + offset)
			img_crop = img[y_min:y_max, x_min:x_max]
			normalize_image(img_crop, img_white, y_max - y_min, x_max - x_min)
		else:
			img_crop = img[y-offset:y + h+offset, x-offset:x + w+offset]
			normalize_image(img_crop, img_white, h, w)
		
		#This displays the cropped/normalized image
		if img_crop is not None and img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
			cv2.imshow("ImageCrop", img_crop)
			cv2.imshow("ImageWhite", img_white)
			prediction, index = classifier.getPrediction(img_white)
			print(labels_dict[index])
			label_list.append(labels_dict[index])
			counter += 1
			
			if counter == 30:
				most_common_label = max(set(label_list), key=label_list.count)
				print("TYPING: " + most_common_label)
				if most_common_label == 'BACKSPACE':
					pyautogui.press('backspace')
				elif most_common_label == 'SPACE':
					pyautogui.press('space')
				elif most_common_label == 'NEWLINE':
					pyautogui.press('enter')
				elif most_common_label == 'DOUBLEQUOTE':
					pyautogui.press('"')
				elif most_common_label == 'EQUALS':
					pyautogui.press('=')
				elif most_common_label == 'SEMICOLON':
					pyautogui.press(';')
				elif most_common_label == 'PLUS':
					pyautogui.press('+')
				elif most_common_label == 'LESSTHAN':
					pyautogui.press('<')
				elif most_common_label == 'PRINT':
					pyautogui.write("System.out.println(")
				else:
					pyautogui.write(most_common_label.lower())
				label_list.clear()
				counter = 0
	
	cv2.imshow("Image", img)

