import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/Eveline/Documents/Thesis/Camera detection/Face detection programming/Webcam-Face-Detect-master/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Eveline/Documents/Thesis/Camera detection/Face detection programming/Webcam-Face-Detect-master/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('C:/Users/Eveline/Documents/Thesis/Camera detection/Face detection programming/Webcam-Face-Detect-master/haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('C:/Users/Eveline/Documents/Thesis/Camera detection/Face detection programming/Webcam-Face-Detect-master/haarcascade_mcs_mouth.xml')

# Video capture
cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

#startup camera
do = 'left'
detected = 'left'
timer = 30
show = ['left'] * timer

# Save camera recordings
saving_left = cv2.VideoWriter(
    'out_left.avi',     # Filename
    cv2.VideoWriter_fourcc('M','J','P','G'),               # Codec for compression
    20,                                 # Frames per second
    (640, 480),                         # Width / Height tuple
    True                                # Color flag
)
saving_right = cv2.VideoWriter(
    'out_right.avi',     # Filename
    cv2.VideoWriter_fourcc('M','J','P','G'),               # Codec for compression
    20,                                 # Frames per second
    (640, 480),                         # Width / Height tuple
    True                                # Color flag
)
saving_both = cv2.VideoWriter(
	'out_both.avi',     # Filename
	cv2.VideoWriter_fourcc('M','J','P','G'),               # Codec for compression
	20,                                 # Frames per second
	(640, 480),                         # Width / Height tuple
	True                                # Color flag
)


while (cv2.waitKey(1) & 0xFF != ord('q')):
	# Capture frame-by-frame
	ret1, frame_l = cap.read()
	ret2, frame_r = cap2.read()
	camera = 'left'

	features_left = ['left']  # list of features found left camera
	features_right = ['right'] # list of features found right camera

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame_l, 'show left',(10,140), font, 1,(0,0,0),2,cv2.LINE_AA)
	cv2.putText(frame_r,'show right',(10,140), font, 1,(0,0,0),2,cv2.LINE_AA)


	for frame in (frame_l, frame_r): # do feature detection for each camera
		feature_list = []

		# Add legenda
		#cv2.putText(frame,'face',(10,50), font, 1,(255,0,0),2,cv2.LINE_AA)
		#cv2.putText(frame,'eyes',(10,80), font, 1,(0,255,0),2,cv2.LINE_AA)
		#cv2.putText(frame,'nose',(10,110), font, 1,(0,0,255),2,cv2.LINE_AA)
		#cv2.putText(frame,str(detected),(10,170), font, 1,(0,0,255),2,cv2.LINE_AA)
		#cv2.putText(frame,'show',(10,200), font, 1,(0,0,255),2,cv2.LINE_AA)
		#cv2.putText(frame,'mouth',(10,140), font, 1,(0,0,0),2,cv2.LINE_AA)

		# Face detection
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 6)

		# add number of faces found
		if len(faces) == None:
			feature_list.append(0)
		else:
			feature_list.append(len(faces))

		#draw rectangle around faces found
		for (x, y, w, h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

			# Feature detection in face
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]

			eyes = eye_cascade.detectMultiScale(roi_gray,2,6, 1)
			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				#cv2.circle(roi_color,(ex+int(0.5*ew),ey+int(0.5*eh)), 2, (0,255,0), -1)

			nose = nose_cascade.detectMultiScale(roi_gray, 1.1,6,1)
			for (nx,ny,nw,nh) in nose:
				cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)


			#mouth = mouth_cascade.detectMultiScale(roi_gray, 2, 6, 1)
			#for (mx,my,mw,mh) in mouth:
				#cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,0),2)

			#Count detected features
			for feature in (eyes, nose):
				if len(feature) == None:
					coordinates = np.array(feature).tolist()
					feature_list.append(len(coordinates))

		# Make feature detection list
		for i in feature_list:
			if camera == 'left':
				features_left.append(i)
			elif camera == 'right':
				features_right.append(i)

		#next feature detection round for other camera
		camera = 'right'

	count = 0

	# Camera decision
	if features_left[1] >= 1 and features_right[1] >= 1: # faces in both cameras?
		do = show[-1] 			#show the one just showed
		detected = 'both faces'
		show.append(do)

	elif features_left[1] >= 1 and features_right[1] == 0: # face in left camera?
		for i in range(1,timer):  # just switched? --> no
			if show[-i] == show[-i-1]:
				count += 1
		if count >= timer-1:
			do = 'left'   # allowed to switch
			show.append('left')
		else:   # just switched --> yes
			do = show[-1]  # pick camera just showed
			show.append(do)
		detected = 'one face, left'

	elif features_right[-1] >= 1 and features_left[1] == 0: #face in right camera?
		for i in range(1,timer):  # just switched? --> no
			if show[-i] == show[-i-1]:
				count += 1
		if count >= timer-1:
			do = 'right'   # allowed to switch
			show.append('right')
		else:   # just switched --> yes
			do = show[-1]
			show.append(do)
		detected = 'one face, right'

	else:
		detected = 'no face detected'
		do = show[-1]
		show.append(do)

	show = show[1:]  # remove the first one in the list, to keep the data in a specific range


	# show camera
	if do == 'left':
		cv2.imshow('both', frame_l)
		#saving_both.write(frame_l)  #save the frame for the 'combined' recording
	elif do == 'right':
		cv2.imshow('both', frame_r)
		#saving_both.write(frame_r) #save the frame for the 'combined' recording

	#cv2.imshow('left', frame_l)
	#cv2.imshow('right', frame_r)
	#saving_left.write(frame_l)
	#saving_right.write(frame_r)


cap.release()
cap2.release()
cv2.destroyAllWindows()
