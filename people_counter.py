from numpy.lib.shape_base import _put_along_axis_dispatcher
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from mylib import thread
from imutils.video import VideoStream
import numpy as np
import time, dlib, cv2, imutils, redis, struct, requests, json, argparse, threading

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-u", "--url", type=str,
                help="url for http host")
ap.add_argument("-i", "--input", type=str,
                help="input for IP camera")
ap.add_argument("-r", "--redis", type=str,
                help="input for redis host")
args = vars(ap.parse_args())

Thread = True
r = redis.Redis(host=args["redis"], port=6379, db=0)

def run(input):
	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	vs = VideoStream(input).start()
	time.sleep(2.0)

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None

	isPosted = False

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=100, maxDistance=100)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]

	host_url = args["url"]
	headers = {
            'Content-Type': 'application/json'
        }
	
	if Thread:
		vs = thread.ThreadingClass(input)


	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width = 640)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % 30 == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > 0.4:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (0, (H // 2) + 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:
					if direction:
						if centroid[1] < H // 2:
							to.stepA = True
						else:
							to.stepB = True
					
					if to.stepA and to.stepB: 
						if direction < 0 and centroid[1] < H // 2:
							totalUp += 1
							empty.append(totalUp)
							to.counted = True
						elif direction > 0 and centroid[1] > H // 2:
							totalDown += 1
							empty1.append(totalDown)
							to.counted = True
						
						x = []
						# compute the sum of total people inside
						x.append(len(empty1)-len(empty))
						#print("Total people inside:", x)


			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# increment the total number of frames processed thus far and
		totalFrames += 1

		if status == "Waiting" and not isPosted:
			x = threading.Thread(target=requestAPI, args=(totalUp, totalDown, empty1, empty, host_url, headers), daemon=True)
			x.start()
			isPosted = True
		elif status == "Tracking" and isPosted:
			isPosted = False

		toRedis(r, frame, 'image')	

def toRedis(r, a, n):
   """Store given Numpy array 'a' in Redis under key 'n'"""
   h, w = a.shape[:2]
   shape = struct.pack('>II', h, w)
   encoded = shape + a.tobytes()

   # Store encoded data in Redis
   r.set(n, encoded)
   return

def requestAPI(totalUp, totalDown, empty1, empty, host_url, headers):
	try:
		payload = json.dumps({
						"enter": totalDown,
						"exit": totalUp,
						"total": len(empty1)-len(empty)
					})
		response = requests.request("POST", host_url, headers=headers, data=payload)
		print(response.json())
	except:
		response = requests.request("POST", host_url, headers=headers, data=payload)
		print(response.json())

run(args["input"])
