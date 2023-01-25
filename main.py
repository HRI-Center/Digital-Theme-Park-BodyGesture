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


# socket 통신 IP 주소
HOST = '192.168.0.31'
PORT = 9999

# 각 joint 각도 계산 함수.
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
        self.detect_ratio = round(160 / 280 * math.tan(math.radians(45)),4)
        self.detect_divider = 10
        self.lock = threading.Lock()
        self.socket_enable = socket_enable
        self.rxflag = True
        self.presendmotion = ''
        self.humanloc = 0
        self.run_video = True
        self.max_motion = 'default'

        # socket enable 은 class 생성시 코드를 수정하여 개발자가 on-off
        if self.socket_enable:
            self.clientSock = socket(AF_INET, SOCK_STREAM)
            self.clientSock.connect((HOST, PORT))

    def send(self, sock,senddata = 'test data'):
        sock.send(senddata.encode())
        print(senddata)

    def receive(self, sock):

        while self.socket_enable:
            recvdata = sock.recv(1024)      # 1024 버퍼 크기 데이터 receive
            msg = recvdata.decode('utf-8')  # byte 데이터 decode
            print('받은 데이터:', msg)

            # kinectic display로 부터 motion end, P2P End 신호를 받으면 rxflag set
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
                ret2, depth_image = capture.get_transformed_depth_image() # get depth image [unit : mm]
                ret3, color_depth_image = capture.get_transformed_colored_depth_image() # get colored depth image
            except:
                continue
            time.sleep(0.01) # waiting for kinetic
            self.lock.acquire()  # thread mutex lock for synchronizing variables with main thread
            # RGB 데이터가 문제 없다면 class 변수에 copy
            if ret1:
                self.image = image
            # Depth 데이터가 문제 없다면 class 변수에 copy
            if ret2:
                self.depth_image = depth_image.astype('int') # type casting unsigned short[raw data from azure kinect] to int
            # colored depth(visualization용) 데이터가 문제 없다면 class 변수에 copy
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
        t.start() # thread start

        self.cur = time.time()
        self.pre = self.cur

        # motion counting dictionary
        motion_dict = {"up": 0, "down" : 0, "left" : 0, "right" : 0, "push" : 0, "pull" : 0}
        self.cur_modetime = time.time() # Timer variable (P2P mode <----N seconds----> Body gesture mode)
        self.pre_modetime = self.cur_modetime
        self.poseflag = False # if poseflag is set, Body gesture mode starts

        # Send message("start! Hi") to kinect controller to test socket I/O at first.
        if self.socket_enable:
            self.send(sock=self.clientSock, senddata="start! Hi")

        while True:
            self.cur_modetime = time.time()
            if self.image is None:
                continue

            # Stream thread로부터 3가지 이미지 데이터 (RGB, depth, colored_depth) 3가지 가져옴.
            self.lock.acquire() # 쓰레드 간 데이터 충돌 방지용 lock
            color_image = self.image # RGB 데이터
            depth_image = self.depth_image # Depth 데이터 각 Pixel 값은 mm 단위
            color_depth_image = self.color_depth_image # 테스트를 위한 Depth 값 시각화 용
            self.lock.release()

            # get frame size of image
            height, width, _ = color_image.shape

            # width of range 1 ~ 10 lines for tracking location of human
            detect_width = int(round(self.detect_ratio * width))
            results = pose.process(color_image) # calculation joint from RGB-image data using MediaPipe

            if results.pose_landmarks:

                # Draw the pose annotation on the image.
                landmarks = results.pose_landmarks.landmark

                # get x, y values of each joints
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

                # --- calcuation of angle of each joints-----
                shoulderLtheta = int(calculate_angle(leftElbow, leftShoulder,leftHip))
                shoulderRtheta = int(calculate_angle(rightElbow, rightShoulder, rightHip))
                elbowLtheta = int(calculate_angle(leftWrist, leftElbow, leftShoulder))
                elbowRtheta = int(calculate_angle(rightWrist, rightElbow, rightShoulder))


                # -----Display each joints angle
                cv2.putText(color_image, str(elbowLtheta),
                            tuple(np.multiply(leftElbow, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                            ) # left elbow joint angle

                cv2.putText(color_image, str(shoulderLtheta),
                            tuple(np.multiply(leftShoulder, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                            ) # left shoulder joint angle

                cv2.putText(color_image, str(elbowRtheta),
                            tuple(np.multiply(rightElbow, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                            ) # right elbow joint angle

                cv2.putText(color_image, str(shoulderRtheta),
                            tuple(np.multiply(rightShoulder, [width, height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                            ) # right shoulder joint angle

                try: # use 'try' sentence in case that each joint has null value(null value is obtained when the joint is not detected on frame)

                    # Mediapipe 에서 검출된 각 joint pixel 좌표계 --->Pixel 좌표계 1280x720 (x,y)로 변환
                    # 'joint'_pos[0] : 픽셀상 x 좌표, 'joint'_pos[1] : 픽셀상 y 좌표
                    # 단, 'joint'_depth는 Real-World 좌표계 mm 단위를 사용 (Azure kinect로 부터 각 joint까지 z축 거리)

                    nose_pos = tuple(np.multiply(nose, [width, height]).astype(int))
                    lefthand_pos = tuple(np.multiply(leftWrist, [width, height]).astype(int))
                    lefthand_depth = depth_image[lefthand_pos[1]][lefthand_pos[0]]
                    righthand_pos = tuple(np.multiply(rightWrist, [width, height]).astype(int))
                    righthand_depth = depth_image[righthand_pos[1]][righthand_pos[0]]
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

                    # 사람 추정 중심 좌표 : 왼쪽 어깨 오른쪽 어깨의 중앙 값을 인식된 사람의 중심 좌표 값으로 봄.
                    x = int((leftShoulder_pos[0] + rightShoulder_pos[0]) / 2)
                    y = int((leftShoulder_pos[1] + rightShoulder_pos[1]) / 2)


                    # 사람 위치 데이터는 10 구역으로 나누어 판별 후 Kinectic Display에 전송
                    if x > (width + detect_width)/2:
                        self.humanloc = self.detect_divider # self.detect_divider = 10 / 10 구역으로 설정됨.
                    elif x < (width - detect_width)/2:
                        self.humanloc = 1
                    else:
                        # human loc x 좌표 값 1 ~ 10 사이로 normalization
                        self. humanloc = (x - (width - detect_width)/2) / detect_width * (self.detect_divider-1) + 1
                        self.humanloc = int(self.humanloc)

                    # 좌표 값 Display
                    cv2.putText(color_image, "Human Coordination:" + str(self.humanloc),
                                (width // 10, height // 10 * 9),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.fontcolor, 2, cv2.LINE_AA)

                    # 사람 중심 좌표에 파란색 원으로 그려 시각화
                    color_image = cv2.circle(color_image, (x, y), 10, (255, 255, 0), -1, cv2.LINE_AA)


                    # 모드 전환을 위한 녹색 구간 판별

                    if self.humanloc < self.detect_divider // 2 + 2 and self.humanloc > self.detect_divider // 2 - 2:

                        # 녹색 구간내 들어올 경우 timer 시작
                        self.cur = time.time()
                        # 녹색 구간 내 5초 이상인 경우 motion tracking 모드 on
                        if self.cur_modetime - self.pre_modetime > 5:
                            self.poseflag = True

                        # motion tracking 모드 on이 되었을 경우
                        if self.poseflag:
                            # 각 관절 Drawing
                            mp_drawing.draw_landmarks(
                                color_image,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                            # Both hands up 판별 조건

                            #   1) 양팔 각도 130도 이상
                            #   2) 어깨 joint depth값과 손 joint depth값의 차이가 300mm 이하
                            #   3) 양 손이 눈 위치보다 높이 있어야함. (y 좌표 값이 작을 수록 높이 있음)
                            if shoulderLtheta > 130 and shoulderRtheta > 130\
                                and (abs( leftShoulder_depth - lefthand_depth ) < 300)\
                                and (abs( rightShoulder_depth - righthand_depth ) < 300)\
                                and lefthand_pos[1] < leftEye_pos[1]\
                                and righthand_pos[1] < rightEye_pos[1]:

                                self.motionstate = "up"
                                motion_dict[self.motionstate] += 1
                                cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            # Left arm wave 판별 조건

                            #   1) 왼쪽 팔꿈치 100 도 이상
                            #   2) 왼쪽 어깨 각도 50도 이상 130도 이하
                            #   3) 오른쪽 어깨 각도 70도 이하
                            elif (elbowLtheta > 100) and (shoulderLtheta > 50) and (shoulderLtheta < 130) \
                                and (abs(leftShoulder_depth - lefthand_depth) < 300)\
                                and shoulderRtheta < 70:
                                self.motionstate = "left"
                                motion_dict[self.motionstate] += 1
                                cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            # Right arm wave 판별 조건

                            #   1) 오른쪽 팔꿈치 100 도 이상
                            #   2) 오른쪽 어깨 각도 50도 이상 130도 이하
                            #   3) 왼쪽 어깨 각도 70도 이하
                            elif (elbowRtheta > 100) and (shoulderRtheta > 50) and (shoulderRtheta < 130) \
                                and (abs(rightShoulder_depth - righthand_depth) < 300)\
                                and shoulderLtheta < 70:
                                self.motionstate = "right"
                                motion_dict[self.motionstate] += 1
                                cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            # Pull 판별 조건

                            #   1) 양 손이 몸체 밖으로 350mm 이상 앞으로 뻗어 나와야함. (몸체는 양쪽 골반을 기준 값으로 봄)
                            #   2) 양손의 높이는 코보다는 아래 옆구리보다는 위에 있어야함.

                            elif (rightHip_depth - righthand_depth > 350) and (leftHip_depth - lefthand_depth > 350)\
                                and lefthand_pos[1] > nose_pos[1] and nose_pos[1] > rightEye[1]\
                                and lefthand_pos[1] < (leftShoulder_pos[1] + leftHip_pos[1]) / 2 \
                                and righthand_pos[1] < (rightShoulder_pos[1] + rightHip_pos[1]) / 2:
                                self.motionstate = "pull"
                                motion_dict[self.motionstate] += 1
                                #print(rightHip_depth - righthand_depth, leftHip_depth - lefthand_depth)
                                cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            # Down 판별 조건

                            #   1) 양 손이 몸체로 부터 100 ~ 350mm 사이 앞으로 위치해 있어야함.
                            #   2) 양 팔꿈치의 각도는 90도 이하.
                            #   3) 바로 직전에 취한 동작이 Both hands up이어야 함. (약속된 kinetic display 모션 순서 up --> down)
                            elif ( 100 < rightShoulder_depth - righthand_depth < 350) and ( 100 < leftShoulder_depth - lefthand_depth < 350) \
                                    and elbowLtheta > 90 and elbowRtheta > 90 :
                                    #and  leftShoulder_pos[1] < lefthand_pos[1] < leftHip_pos[1] \
                                    #and  rightShoulder_pos[1] < righthand_pos[1] < rightHip_pos[1]:
                                if self.prestate == "up" or self.motionstate == "down":
                                    self.motionstate = "down"
                                    motion_dict[self.motionstate] += 1
                                    cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            # Push 판별 조건

                            #   1) 양 손이 몸체로 부터 200mm 이내 위치
                            #   2) 양 손의 높이는 어깨보다 아래에 위치
                            #   3) 양 팔꿈치는 70도 이하로 구부리기
                            #   4) 바로 직전에 취한 동작이 pull이여야함. (약속된 kinetic display 모션 순서 pull --> push)
                            elif (rightShoulder_depth - righthand_depth < 200) and (leftShoulder_depth - lefthand_depth < 200) \
                                and (lefthand_pos[1] > leftShoulder_pos[1]) and elbowLtheta < 70 \
                                and (righthand_pos[1] > rightShoulder_pos[1]) and elbowRtheta < 70:

                                if self.prestate == 'pull' or self.motionstate == "push":
                                    self.motionstate = "push"
                                    motion_dict[self.motionstate] += 1
                                    cv2.putText(color_image, self.motionstate, (width // 10, height // 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, self.fontsize, self.fontcolor, 2, cv2.LINE_AA)

                            # kinetic display에 보내는 통신 주기 용 timer 변수 interval
                            interval = self.cur - self.pre

                            # Motion은 1초에 한번씩만 kinetic display로 보내기로 약속됨.
                            if interval > 1:
                                self.pre = self.cur

                                # 1초 동안 가장 많이 detect된 모션을 kinetic display로 전송
                                max_motion = max(motion_dict,key=motion_dict.get)
                                print("Count : ", motion_dict)

                                # 가장 많이 counting 된 동작 또한 1초 내 최소 15번 이상 감지되어야 해당 동작을 취했다고 판별.
                                if motion_dict[max_motion] > 15:

                                    # rxflag는 kinect display로 부터 준비 신호를 받을 때 set 됨.
                                    if self.socket_enable and self.rxflag:
                                        sendmotion = 'mexecute1 '+max_motion  # protocol : mexecute1 + 모션
                                        
                                        # 이전에 보낸 motion이 현재 motion과 같으면 중복 해서 보내지 않는다.
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
                            
                            # push, down 판별을 위한 변수
                            if self.prestate != self.motionstate:
                                self.prestate = self.motionstate
                            # ex) prestate 가 up 이고 현재 motionstate가 down 판별 조건을 만족하면 down으로 인식
                    else:
                        self.pre_modetime = self.cur_modetime
                        self.poseflag = False
                        
                    # Motion tracking 모드 off, only 사람 위치 추정
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

            # 화면 상 1 ~ 10 구역 표시를 위한 코드

            # (프레임 가로 길이) - (1 ~ 10구역 영역 가로 길이)의 절반
            startline = (width - detect_width) // 2

            for i in range(self.detect_divider):
                # 3, 7 구역을 녹색으로 drawing
                if i == (self.detect_divider // 2 - 2)  or i == (self.detect_divider // 2 + 2):
                    cv2.line(color_image,(startline + detect_width//self.detect_divider*i,0), (startline + detect_width//self.detect_divider*i,height), (10,200,10), 2)
                # 나무저 구역 회색으로 drawing
                else:
                    cv2.line(color_image,(startline + detect_width//self.detect_divider*i,0), (startline + detect_width//self.detect_divider*i,height), (150, 150, 150), 1)

            # 이미지 display
            cv2.imshow('colored depth image', color_depth_image)
            cv2.imshow('color image', color_image)

            # ESC key 입력
            if cv2.waitKey(10) & 0xFF == 27:
                self.lock.acquire()
                self.run_video = False      # stream video thread 종료
                self.socket_enable = False  # socket thread 종료
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


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
