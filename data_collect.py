import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
img_size = 300
counter = 0
start_countdown = False

#Change to whatever label you want to collect images to at the current moment
img_folder = "data/Statements/PRINT"


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

def show_countdown(img, count):
	font = cv2.FONT_HERSHEY_SIMPLEX 
	cv2.putText(img, str(count), (img.shape[1]//2, img.shape[0]//2), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow("Image", img)
	time.sleep(1)

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
	
	# Check if we should start the countdown
	if start_countdown:
		if countdown_time > 0:
			show_countdown(img, countdown_time)
			countdown_time -= 1
		else:
			counter += 1
			cv2.imwrite(f'{img_folder}/image_{time.time()}.jpg', img_white)
			print(f'saved image: {counter}')
			if counter >= 135:
				start_countdown = False
				counter = 0
	else:
		cv2.imshow("Image", img)
	key = cv2.waitKey(1)
	if key == ord(" "):
		start_countdown = True
		countdown_time = 5

