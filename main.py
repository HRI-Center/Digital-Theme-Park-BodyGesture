import sys
import threading

import cv2
import numpy as np
import math
import mediapipe as mp
from yolo import *
from threading import Thread
sys.path.insert(1, '../')
import pykinect_azure as pykinect
import time
from socket import *
from socket_comm import *

HOST = '192.168.0.31'
#HOST = '192.168.11.220'
#HOST = '192.168.0.30'
PORT = 9999

def calculate_angle(a: object, b: object, c: object) -> object:
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

class Kinect:
    def __init__(self,body_tracking,socket_enable = False):

        # Initialize the library, if the library is not found, add the library path as argument
        pykinect.initialize_libraries(track_body=body_tracking)

        # Modify camera configuration
        self.device_config = pykinect.default_configuration
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

        # Start device
        self.device = pykinect.start_device(config=self.device_config)

        # Start body tracker
        if body_tracking:
            self.bodyTracker = pykinect.start_body_tracker()

        self.depth_image = None
        self.color_depth_image = None
        self.image = None
        self.motionstate = 'default'
        self.prestate = self.motionstate
        self.fontcolor = (0,100,200)
        self.fontsize = 2
        self.colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.is_cuda = True
        self.net = build_model(self.is_cuda)
        self.detect_ratio = round(160 / 280 * math.tan(math.radians(45)),4)
        self.detect_divider = 10
        self.lock = threading.Lock()
        self.socket_enable = socket_enable
        self.rxflag = True
        self.presendmotion = ''
        self.humanloc = 0
        self.run_video = True
        self.max_motion = 'default'


        if self.socket_enable:
            self.clientSock = socket(AF_INET, SOCK_STREAM)
            self.clientSock.connect((HOST, PORT))

    def send(self, sock,senddata = 'test data'):
        sock.send(senddata.encode())
        print(senddata)

    def receive(self, sock):

        while self.socket_enable:
            recvdata = sock.recv(1024)
            #if not recvdata:
            #    #print('no receive data')
            #    sock.close()
            #    break
            #else:
            msg = recvdata.decode('utf-8')
            print('받은 데이터:', msg)
            if self.poseflag:
                if 'Motion End' in msg:
                    self.lock.acquire()
                    self.rxflag = True
                    self.lock.release()
            else:
                if 'P2P End' in msg:
                    self.lock.acquire()
                    self.rxflag = True
                    self.lock.release()

            time.sleep(1)
        sock.close()

            #self.send_func(sock)

    def send_func(self, sock):
        #temp = input("전송대기중....")
        senddata = 'ok'
        sock.send(senddata.encode('utf-8'))
        print('전송완료')

    def run(self):
        receiver = threading.Thread(target=self.receive, args=(self.clientSock,))
        receiver.daemon = True
        receiver.start()

    def run_kinect(self):

        cv2.namedWindow('Color image with skeleton', cv2.WINDOW_NORMAL)
        while True:
            # Get capture
            capture = self.device.update()

            # Get body tracker frame
            body_frame = self.bodyTracker.update()

            # Get the color image
            ret, color_image = capture.get_color_image()
            _, depth_image = capture.get_colored_depth_image()
            if not ret:
                print("fail to get frame")
                continue

            # Draw the skeletons into the color image
            color_skeleton = body_frame.draw_bodies(color_image, pykinect.K4A_CALIBRATION_TYPE_COLOR)

            # Overlay body segmentation on depth image
            cv2.imshow('Color image with skeleton', color_skeleton)

            # Press q key to stop
            if cv2.waitKey(1) == 27:
                break
    # Mouse click event for checking pixel coordination in opencv window
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            print(x, y, self.depth_image[y][x])

    # Video I/O is running in a single thread module
    def stream_video(self):

        print("Start Kinect video thread")
        while self.run_video: # running flag
            try:
                capture = self.device.update() # get kinect image datas
                ret1, image = capture.get_color_image() # get RGB frame
                # image = cv2.flip(image,1)
                ret2, depth_image = capture.get_transformed_depth_image() # get depth image [unit : mm]
                ret3, color_depth_image = capture.get_transformed_colored_depth_image() # get colored depth image
            except:
                continue
            time.sleep(0.01) # waiting for kinetic
            self.lock.acquire()  # thread mutex lock for synchronizing variables with main thread
            if ret1:
                self.image = image
            if ret2:
                self.depth_image = depth_image.astype('int')
            if ret3:
                self.color_depth_image = color_depth_image
            self.lock.release() # thread mutex unlock
        self.device.close()

        print("kinect video thread is terminated.")

    def run_mediapipe(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        cv2.namedWindow('colored depth image')
        cv2.setMouseCallback('colored depth image', self.mouse_event)

        # socket mode connected with kinetic controller through TCP/IP
        if self.socket_enable:
            # create a thread to receive message from kinetic controller
            receiver = Thread(target=self.receive, args=(self.clientSock,))
            receiver.daemon = True
            receiver.start() # thread start

        # kinect thread start
        t = Thread(target = self.stream_video)
        t.start()

        self.cur = time.time()
        self.pre = self.cur

        # motion counting dictionary
        motion_dict = {"up": 0, "down" : 0, "left" : 0, "right" : 0, "push" : 0, "pull" : 0}
        self.cur_modetime = time.time() # Timer variable (P2P mode <----N seconds----> Body gesture mode)
        self.pre_modetime = self.cur_modetime
        self.poseflag = False # if poseflag is set, Body gesture mode starts
        if self.socket_enable:
            self.send(sock=self.clientSock, senddata="start! Hi")
        while True:
            self.cur_modetime = time.time()
            if self.image is None:
                continue

            # thread mutex lock to get image data from azure kinect thread
            self.lock.acquire()
            color_image = self.image
            depth_image = self.depth_image
            color_depth_image = self.color_depth_image
            self.lock.release()

            height, width, _ = color_image.shape
            detect_width = int(round(self.detect_ratio * width))
            results = pose.process(color_image)

            if results.pose_landmarks:
                # Draw the pose annotation on the image.
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                landmarks = results.pose_landmarks.landmark
                nose = [landmarks[mp_pose.PoseLandmark.NOSE].x,landmarks[mp_pose.PoseLandmark.NOSE].y]
                leftEye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
                rightEye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]

                leftHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                leftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                leftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                rightHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                rightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                shoulderLtheta = int(calculate_angle(leftElbow, leftShoulder,leftHip))
                shoulderRtheta = int(calculate_angle(rightElbow, rightShoulder, rightHip))
                elbowLtheta = int(calculate_angle(leftWrist, leftElbow, leftShoulder))
                elbowRtheta = int(calculate_angle(rightWrist, rightElbow, rightShoulder))

                cv2.putText(color_image, str(elbowLtheta),
                            tuple(np.multiply(leftElbow, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                            )

                cv2.putText(color_image, str(shoulderLtheta),
                            tuple(np.multiply(leftShoulder, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                            )

                cv2.putText(color_image, str(elbowRtheta),
                            tuple(np.multiply(rightElbow, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                            )

                cv2.putText(color_image, str(shoulderRtheta),
                            tuple(np.multiply(rightShoulder, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                            )

                try:

                    nose_pos = tuple(np.multiply(nose, [width, height]).astype(int))
                    lefthand_pos = tuple(np.multiply(leftWrist, [width, height]).astype(int))
                    lefthand_depth = depth_image[lefthand_pos[1]][lefthand_pos[0]]
                    cv2.putText(color_image, str(lefthand_depth), lefthand_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1,
                                cv2.LINE_AA)

                    righthand_pos = tuple(np.multiply(rightWrist, [width, height]).astype(int))
                    righthand_depth = depth_image[righthand_pos[1]][righthand_pos[0]]
                    cv2.putText(color_image, str(righthand_depth), righthand_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1,
                                cv2.LINE_AA)

                    leftEye_pos = tuple(np.multiply(leftEye, [width, height]).astype(int))
                    rightEye_pos = tuple(np.multiply(rightEye, [width, height]).astype(int))
                    leftShoulder_pos = tuple(np.multiply(leftShoulder, [width, height]).astype(int))
                    rightShoulder_pos = tuple(np.multiply(rightShoulder, [width, height]).astype(int))
                    leftHip_pos = tuple(np.multiply(leftHip, [width, height]).astype(int))
                    rightHip_pos = tuple(np.multiply(rightHip, [width, height]).astype(int))

                    leftShoulder_depth = depth_image[leftShoulder_pos[1]][leftShoulder_pos[0]]
                    cv2.putText(color_image, str(leftShoulder_depth), leftShoulder_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1,
                                cv2.LINE_AA)

                    rightShoulder_depth = self.depth_image[rightShoulder_pos[1]][rightShoulder_pos[0]]
                    cv2.putText(color_image, str(rightShoulder_depth), rightShoulder_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1,
                                cv2.LINE_AA)

                    leftHip_depth = depth_image[leftHip_pos[1]][leftHip_pos[0]]
                    cv2.putText(color_image, str(leftHip_depth), leftHip_pos, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (200, 200, 200), 1, cv2.LINE_AA)
                    rightHip_depth = depth_image[rightHip_pos[1]][rightHip_pos[0]]
                    cv2.putText(color_image, str(leftHip_depth), rightHip_pos, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (200, 200, 200), 1, cv2.LINE_AA)

                    x = int((leftShoulder_pos[0] + rightShoulder_pos[0]) / 2)
                    y = int((leftShoulder_pos[1] + rightShoulder_pos[1]) / 2)

                    if x > (width + detect_width)/2:
                        self.humanloc = self.detect_divider
                    elif x < (width - detect_width)/2:
                        self.humanloc = 1
                    else:
                        self. humanloc = (x - (width - detect_width)/2) / detect_width * (self.detect_divider-1) + 1
                        self.humanloc = int(self.humanloc)

                    cv2.putText(color_image, "Human Coordination:" + str(self.humanloc),
                                (width // 10, height // 10 * 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.fontcolor, 2, cv2.LINE_AA)
                    color_image = cv2.circle(color_image, (x, y), 10, (255, 255, 0), -1, cv2.LINE_AA)

                    if self.humanloc < self.detect_divider // 2 + 2 and self.humanloc > self.detect_divider // 2 - 2:
                        self.cur = time.time()
                        if self.cur_modetime - self.pre_modetime > 5:
                            self.poseflag = True
                            self.send(sock=self.clientSock, senddata='init')

                        if self.poseflag:
                            mp_drawing.draw_landmarks(
                                color_image,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                            # Both hands up
                            if shoulderLtheta > 130 and shoulderRtheta > 130\
                                and (abs( leftShoulder_depth - lefthand_depth ) < 300)\
                                and (abs( rightShoulder_depth - righthand_depth ) < 300)\
                                and lefthand_pos[1] < leftEye_pos[1]\
                                and righthand_pos[1] < rightEye_pos[1]:
                                self.motionstate = "up"
                                motion_dict[self.motionstate] += 1
                                cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            # Left arm wave
                            elif (elbowLtheta > 100) and (shoulderLtheta > 50) and (shoulderLtheta < 130) \
                                and (abs(leftShoulder_depth - lefthand_depth) < 300)\
                                and shoulderRtheta < 70:
                                self.motionstate = "left"
                                motion_dict[self.motionstate] += 1
                                cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            # Right arm wave
                            elif (elbowRtheta > 100) and (shoulderRtheta > 50) and (shoulderRtheta < 130) \
                                and (abs(rightShoulder_depth - righthand_depth) < 300)\
                                and shoulderLtheta < 70:
                                self.motionstate = "right"
                                motion_dict[self.motionstate] += 1
                                cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            elif (rightHip_depth - righthand_depth > 350) and (leftHip_depth - lefthand_depth > 350)\
                                and lefthand_pos[1] > nose_pos[1] and nose_pos[1] > rightEye[1]\
                                and lefthand_pos[1] < (leftShoulder_pos[1] + leftHip_pos[1]) / 2 \
                                and righthand_pos[1] < (rightShoulder_pos[1] + rightHip_pos[1]) / 2:
                                self.motionstate = "pull"
                                motion_dict[self.motionstate] += 1
                                #print(rightHip_depth - righthand_depth, leftHip_depth - lefthand_depth)
                                cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            elif ( 100 < rightShoulder_depth - righthand_depth < 350) and ( 100 < leftShoulder_depth - lefthand_depth < 350) \
                                    and elbowLtheta > 90 and elbowRtheta > 90 :
                                    #and  leftShoulder_pos[1] < lefthand_pos[1] < leftHip_pos[1] \
                                    #and  rightShoulder_pos[1] < righthand_pos[1] < rightHip_pos[1]:
                                if self.prestate == "up" or self.motionstate == "down":
                                    self.motionstate = "down"
                                    motion_dict[self.motionstate] += 1
                                    cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            elif (rightShoulder_depth - righthand_depth < 200) and (leftShoulder_depth - lefthand_depth < 200) \
                                and (lefthand_pos[1] > leftShoulder_pos[1]) and elbowLtheta < 70 \
                                and (righthand_pos[1] > rightShoulder_pos[1]) and elbowRtheta < 70:

                                if self.prestate == 'pull' or self.motionstate == "push":
                                    self.motionstate = "push"
                                    motion_dict[self.motionstate] += 1
                                    cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            interval = self.cur - self.pre
                            if interval > 1:
                                self.pre = self.cur
                                max_motion = max(motion_dict,key=motion_dict.get)
                                print("Count : ", motion_dict)
                                if motion_dict[max_motion] > 15:

                                    if self.socket_enable and self.rxflag:
                                        sendmotion = 'mexecute1 '+max_motion
                                        if self.presendmotion != sendmotion:

                                            print("Send : ", max_motion)
                                            self.send(sock=self.clientSock, senddata=sendmotion)
                                            self.presendmotion = sendmotion
                                        else:
                                            print('same motion with previous motion')
                                        self.lock.acquire()
                                        self.rxflag = False
                                        self.lock.release()
                                for key in motion_dict.keys():
                                    motion_dict[key] = 0

                            if self.prestate != self.motionstate:
                                self.prestate = self.motionstate

                    else:
                        self.pre_modetime = self.cur_modetime
                        self.poseflag = False
                    if not self.poseflag:
                        sendpose = 'pexecute ' + str(self.humanloc)
                        print('Send : ', sendpose)
                        if self.socket_enable and self.rxflag:
                            self.send(sock=self.clientSock, senddata=sendpose)
                            self.lock.acquire()
                            self.rxflag = False
                            self.lock.release()
                except:
                    pass
            startline = (width - detect_width) // 2

            for i in range(self.detect_divider):
                if i == (self.detect_divider // 2 - 2)  or i == (self.detect_divider // 2 + 2):
                    cv2.line(color_image,(startline + detect_width//self.detect_divider*i,0), (startline + detect_width//self.detect_divider*i,height), (10,200,10), 2)
                else:
                    cv2.line(color_image,(startline + detect_width//self.detect_divider*i,0), (startline + detect_width//self.detect_divider*i,height), (150, 150, 150), 1)

            cv2.imshow('colored depth image', color_depth_image)
            cv2.imshow('color image', color_image)

            if cv2.waitKey(10) & 0xFF == 27:
                self.lock.acquire()
                self.run_video = False
                self.socket_enable = False
                self.lock.release()
                t.join()
                if self.socket_enable:
                    receiver.join()
                print("Program is terminated")
                break

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    azure = Kinect(body_tracking=False, socket_enable=True)
    azure.run_mediapipe()

    #azure.stream_video()
    #azure.human_detection()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
