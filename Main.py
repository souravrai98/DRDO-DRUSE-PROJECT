import socket
from threading import Thread
import time

#from gpiozero import DigitalOutputDevice
#from gpiozero import PWMOutputDevice
#from gpiozero import AngularServo

from threading import Lock
#import smbus
#from gps import *
import RPi.GPIO as GPIO
import cv2
import io
import picamera
import logging
import socketserver
from threading import Condition

from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils

IP = "192.10.9.97"
motionCtrlPort = 2110
camCtrlPort = 2111
specialPort = 2222
emergencyPort = 3333
videoPort = 1108
txMotionData=0
txCamData = 0
txSpecialData = 0
txEmergencyData = 0
rxMotionData = 0
rxCamData = 0
rxSpecialData = 0
rxEmergencyData = 0
output=""
camAngle = 0
headAngle = 0
detections = None


stopVideoSteamer = False
stopObjectDetecter = False

stopMotionDataUpdater = False
stopCamDataUpdater = False
stopSpecialDataUpdater = False
stopGpsDataUpdater = False
stopGyroDataUpdater = False
stopUltraSonicDataUpdater = False
stopEmergencyDataUpdater =False

stopSpecialController = False
stopMotionController = False
stopCamController = False


txUltraSonicData = 0
txGyroData = 0
txGPSData = 0

def videoStreamer():
    global output
    from http import server
    while stopVideoSteamer is not True:
        class StreamingOutput(object):
            def __init__(self):
                self.lock = Lock()
                self.frame = io.BytesIO()
                self.clients = []

            def write(self, buf):
                died = []
                if buf.startswith(b'\xff\xd8'):
                    # New frame, send old frame to all connected clients
                    size = self.frame.tell()
                    if size > 0:
                        self.frame.seek(0)
                        data = self.frame.read(size)
                        self.frame.seek(0)
                        with self.lock:
                            for client in self.clients:
                                try:
                                    client.wfile.write(b'--FRAME\r\n')
                                    client.send_header('Content-Type', 'image/jpeg')
                                    client.send_header('Content-Length', size)
                                    client.end_headers()
                                    client.wfile.write(data)
                                    client.wfile.write(b'\r\n')
                                except Exception as e:
                                    died.append(client)
                self.frame.write(buf)
                if died:
                    self.remove_clients(died)

            def flush(self):
                with self.lock:
                    for client in self.clients:
                        client.wfile.close()

            def add_client(self, client):
                print('Adding streaming client %s:%d' % client.client_address)
                with self.lock:
                    self.clients.append(client)

            def remove_clients(self, clients):
                with self.lock:
                    for client in clients:
                        try:
                            print('Removing streaming client %s:%d' % client.client_address)
                            self.clients.remove(client)
                        except ValueError:
                            pass  # already removed

        class StreamingHandler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/live.mjpg':
                    self.close_connection = False
                    self.send_response(200)
                    self.send_header('Age', 0)
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--FRAME')
                    self.end_headers()
                    output.add_client(self)
                else:
                    self.send_error(404)
                    self.end_headers()

        class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
            pass

        print('\nStreaming Started')
        with picamera.PiCamera(resolution='760x420', framerate=30) as camera:
            output = StreamingOutput()
            camera.start_recording(output, format='mjpeg')
            try:
                address = ('',videoPort)
                server = StreamingServer(address, StreamingHandler)
                server.serve_forever()
            finally:
                camera.stop_recording()

def objectDetector():
    from http import server
    global videoPort

    while stopObjectDetecter is not True:

        def classify_frame(net, inputQueue, outputQueue):
            # keep looping
            while True:
                # check to see if there is a frame in our input queue
                if not inputQueue.empty():
                    # grab the frame from the input queue, resize it, and
                    # construct a blob from it
                    frame = inputQueue.get()
                    frame = cv2.resize(frame, (760, 420))
                    blob = cv2.dnn.blobFromImage(frame, 0.007843,
                                                 (760, 420), 127.5)

                    # set the blob as input to our deep learning object
                    # detector and obtain the detections
                    net.setInput(blob)
                    detections = net.forward()

                    # write the detections to the output queue
                    outputQueue.put(detections)

        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--prototxt",
                        help="path to Caffe 'deploy' prototxt file",
                        default='MobileNetSSD_deploy.prototxt.txt')
        ap.add_argument("-m", "--model",
                        help="path to Caffe pre-trained model",
                        default='MobileNetSSD_deploy.caffemodel')
        ap.add_argument("-c", "--confidence", type=float, default=0.2,
                        help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())

        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

        # initialize the input queue (frames), output queue (detections),
        # and the list of actual detections returned by the child process
        inputQueue = Queue(maxsize=1)
        outputQueue = Queue(maxsize=1)


        # construct a child process *indepedent* from our main process of
        # execution
        print("[INFO] starting process...")
        p = Process(target=classify_frame, args=(net, inputQueue,
                                                 outputQueue,))
        p.daemon = True
        p.start()

        class StreamingOutput(object):
            def __init__(self):
                self.frame = None
                self.buffer = io.BytesIO()
                self.condition = Condition()

            def write(self, buf):
                if buf.startswith(b'\xff\xd8'):
                    # New frame, copy the existing buffer's content and notify all
                    # clients it's available
                    self.buffer.truncate()
                    with self.condition:
                        self.frame = self.buffer.getvalue()
                        self.condition.notify_all()
                    self.buffer.seek(0)
                return self.buffer.write(buf)

        class StreamingHandler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/live.mjpg':
                    self.send_response(200)
                    self.send_header('Age', 0)
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                    self.end_headers()
                    try:
                        while True:
                            with output.condition:
                                output.condition.wait()
                                fps = FPS().start()
                                frame = output.frame
                                data = np.fromstring(frame, dtype=np.uint8)
                                frame = cv2.imdecode(data, 1)
                                frame = imutils.resize(frame, width=760)
                                (fH, fW) = frame.shape[:2]

                                # if the input queue *is* empty, give the current frame to
                                # classify
                                if inputQueue.empty():
                                    inputQueue.put(frame)

                                # if the output queue *is not* empty, grab the detections
                                if not outputQueue.empty():
                                    global detections
                                    detections = outputQueue.get()

                                # check to see if our detectios are not None (and if so, we'll
                                # draw the detections on the frame)
                                if detections is not None:
                                    # loop over the detections
                                    for i in np.arange(0, detections.shape[2]):
                                        # extract the confidence (i.e., probability) associated
                                        # with the prediction
                                        confidence = detections[0, 0, i, 2]

                                        # filter out weak detections by ensuring the `confidence`
                                        # is greater than the minimum confidence
                                        if confidence < args["confidence"]:
                                            continue

                                        # otherwise, extract the index of the class label from
                                        # the `detections`, then compute the (x, y)-coordinates
                                        # of the bounding box for the object
                                        idx = int(detections[0, 0, i, 1])
                                        dims = np.array([fW, fH, fW, fH])
                                        box = detections[0, 0, i, 3:7] * dims
                                        (startX, startY, endX, endY) = box.astype("int")

                                        # draw the prediction on the frame
                                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                                                     confidence * 100)
                                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                                      COLORS[idx], 2)
                                        y = startY - 15 if startY - 15 > 15 else startY + 15
                                        cv2.putText(frame, label, (startX, y),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                                        # show the output frame



                                        # update the FPS counter
                                fps.update()
                                fps.stop()
                            # stop the timer and display FPS information

                            r, frame = cv2.imencode(".jpg", frame)

                            self.wfile.write(b'--FRAME\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', len(frame))
                            self.end_headers()
                            self.wfile.write(frame)
                            self.wfile.write(b'\r\n')
                    except Exception as e:
                        logging.warning(
                            'Removed streaming client %s: %s',
                            self.client_address, str(e))
                else:
                    self.send_error(404)
                    self.end_headers()

        class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        with picamera.PiCamera(resolution='426X240', framerate=40)as camera:
            output = StreamingOutput()
            camera.start_recording(output, format='mjpeg')
            try:
                address = ('', videoPort)
                server = StreamingServer(address, StreamingHandler)
                server.serve_forever()
            finally:
                camera.stop_recording()


def motionDataUpdater():
    global IP, motionCtrlPort, rxMotionData
    print('\nmotionDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, motionCtrlPort))
    while stopMotionDataUpdater is not True:
        msg=str(txMotionData)
        sendBytes=msg.encode('utf-8')
        #print("TXMotionData: %f" % txMotionData)
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxMotionData=int(dataString)
        #print("RXMotionData: %f" % rxMotionData)
def camDataUpdater():
    global IP, camCtrlPort, rxCamData
    print('camDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, camCtrlPort))
    while stopCamDataUpdater is not True:
        msg=str(txCamData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxCamData=int(dataString)
def specialDataUpdater():
    global IP, specialPort, rxSpecialData
    #print('specialDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, specialPort))
    while stopSpecialDataUpdater is not True:
        msg=str(txSpecialData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxSpecialData=(dataString)
        #print(rxSpecialData)
def emergencyDataUpdater():
    global IP, emergencyPort, rxEmergencyData, stopEmergencyDataUpdater
    print('emergencyDataUpdater Started')
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((IP, emergencyPort))
    while stopEmergencyDataUpdater is not True:

        msg=str(txEmergencyData)
        sendBytes=msg.encode('utf-8')
        data, addr = sock.recvfrom(1024)
        sock.sendto(sendBytes,addr)
        dataString=data.decode('utf-8')
        rxEmergencyData=int(dataString)

def motionController():
    global txMotionData , camAngle, headAngle

    leftPowerPin = 26  # IN1 - Forward Drive
    leftDirPin = 19  # IN2 - Reverse Drive
    rightPowerPin = 13  # IN1 - Forward Drive
    rightDirPin = 6  # IN2 - Reverse Drive

    leftPower = PWMOutputDevice(leftPowerPin, True, 0, 1000)
    leftDir = DigitalOutputDevice(leftDirPin, True, False)
    rightPower = PWMOutputDevice(rightPowerPin, True, 0, 1000)
    rightDir= DigitalOutputDevice(rightDirPin, True, False)

    #print('\nmotionController Started')
    while stopMotionController is not True:

        power = int(rxMotionData%1000)
       # if power!= 0:
        angle = int(rxMotionData / 1000)
        power = (power + 1) / 256
        if angle in range (80, 101) or angle in range (260, 281):
            rPower = power
            lPower = power
        elif angle in range (101, 181):
            rPower = power
            lPower = power * ((180-angle)/90)
        elif angle in range (181, 261):
            rPower = power
            lPower = power * ((angle - 180) / 90)
        elif angle in range(0, 80):
            rPower = power * ((angle)/90)
            lPower = power
        else:
            rPower = power * ((359-angle) / 90)
            lPower = power

        if lPower<0.1: lPower = 0
        if rPower<0.1: rPower = 0


        if angle in range (0, 181):
            leftDir.off()
            rightDir.off()
            leftPower.value = lPower
            rightPower.value = rPower
        else:
            leftDir.on()
            rightDir.on()
            leftPower.value = 1- lPower
            rightPower.value = 1- rPower
        txMotionData =int(lPower + rPower)
        time.sleep(0.04)
        #print(" rxMD:%d" % rxMotionData, " A:%d" % angle," P:%.2f" % power,
        #      " lPower:%.2f" % lPower, " rPower:%.2f" % rPower)

        #print(" rxMD:%d" % rxMotionData, " txMD:%d" % txMotionData,
        #      " rxCD:%d" % rxCamData, " txCD:%d" % txCamData,
        #     " rxSD:%d" % rxSpecialData, " txSD:%d" % txSpecialData,
        #     " rxED:%d" % rxEmergencyData, " txED:%d" % txEmergencyData)'''
def camController():
    global txCamData, camAngle, headAngle
    camPin = 4
    headPin = 17
    #print('camController Started')
    camServo = AngularServo(camPin, min_angle=0, max_angle=180, min_pulse_width=(0.5 / 1000),
                            max_pulse_width=(2.3 / 1000),frame_width=20/1000)
    headServo = AngularServo(headPin, min_angle=-90, max_angle=90, min_pulse_width=(0.5 / 1000),
                            max_pulse_width=(2.3 / 1000),frame_width=20 / 1000)
    camServo.angle = 90
    headServo.angle = 0
    time.sleep(1)

    while stopCamController is not True:
        txCamData = 0
        #time.sleep(0.500)
        camAngle = int(rxCamData%1000) #end digits  xxxabc  abc is data
        headAngle = int(rxCamData/1000) #begin digits abcxxx
        #if rxCamData:
            #print("rxCamData: %d" % rxCamData)
        camServo.angle = camAngle
        headServo.angle = headAngle
def specialController():
    global txSpecialData
    print('specialController Started')
    while stopSpecialController is not True:
        txSpecialData += 1
        time.sleep(0.1)
        #if rxSpecialData!=0:
        # print("rxSpecialData: %d" % rxSpecialData)
def emergencyController():
    global txEmergencyData , stopMotionController, stopSpecialController, stopCamController, stopMotionDataUpdater, stopCamDataUpdater,stopUltraSonicDataUpdater,stopGyroDataUpdater, stopGpsDataUpdater, stopSpecialDataUpdater, stopEmergencyDataUpdater,stopVideoSteamer, stopObjectDetecter


    lastTxEmergencyData =890

    motionControllerStopCommand = 101
    motionControllerStartCommand = 102

    specialControllerStopCommand = 103
    specialControllerStartCommand = 104

    camControllerStopCommand =105
    camControllerStartCommand = 106

    motionDataUpdaterStopCommand =107
    motionDataUpdaterStartCommand = 108

    camDataUpdaterStopCommand = 109
    camDataUpdaterStartCommand = 110

    ultraSonicDataUpdateStopCommand = 111
    ultraSonicDataUpdateStartCommand =112

    gyroDataUpdaterStopCommand = 113
    gyroDataUpdaterStartCommand = 114

    gpsDataUpdaterStopCommand = 115
    gpsDataUpdaterStartCommand =116

    specialDataUpdaterStopCommand = 117
    specialDataUpdaterStartCommand = 118

    videoSteamerStopCommand = 119
    videoSteamerStartCommand =120

    objectDetectorStopCommand = 121
    objectDetectorStartCommand = 122


    print('emergencyController Started')
    while True:
        if txEmergencyData is not  lastTxEmergencyData:
            lastTxEmergencyData = txEmergencyData
            if txEmergencyData is motionControllerStopCommand:
                stopMotionController = True
            elif txEmergencyData is motionControllerStartCommand:
                stopMotionController = False
            elif txEmergencyData is specialControllerStopCommand:
                stopSpecialController = True
            elif txEmergencyData is specialControllerStartCommand:
                stopSpecialController = False
            elif txEmergencyData is camControllerStopCommand:
                stopCamController = True
            elif txEmergencyData is camControllerStartCommand:
                stopCamController = False
            elif txEmergencyData is motionDataUpdaterStopCommand:
                stopMotionDataUpdater = True
            elif txEmergencyData is motionDataUpdaterStartCommand:
                stopMotionDataUpdater = False
            elif txEmergencyData is camDataUpdaterStopCommand:
                stopCamDataUpdater = True
            elif txEmergencyData is camDataUpdaterStartCommand:
                stopCamDataUpdater = False
            elif txEmergencyData is ultraSonicDataUpdateStopCommand:
                stopUltraSonicDataUpdater = True
            elif txEmergencyData is ultraSonicDataUpdateStartCommand:
                stopUltraSonicDataUpdater = False
            elif txEmergencyData is gyroDataUpdaterStopCommand:
                stopGyroDataUpdater = True
            elif txEmergencyData is gyroDataUpdaterStartCommand:
                stopGyroDataUpdater = False
            elif txEmergencyData is gpsDataUpdaterStopCommand:
                stopGpsDataUpdater = True
            elif txEmergencyData is gpsDataUpdaterStartCommand:
                stopGpsDataUpdater = False
            elif txEmergencyData is specialDataUpdaterStopCommand:
                stopSpecialDataUpdater = True
            elif txEmergencyData is specialDataUpdaterStartCommand:
                stopSpecialDataUpdater = False
            elif txEmergencyData is emergencyDataUpdaterStopCommand:
                stopEmergencyDataUpdater = True
            elif txEmergencyData is emergencyDataUpdaterStartCommand:
                stopEmergencyDataUpdater = False
            elif txEmergencyData is videoSteamerStopCommand:
                stopVideoSteamer = True
            elif txEmergencyData is videoSteamerStartCommand:
                stopVideoSteamer = False
            elif txEmergencyData is objectDetectorStopCommand:
                stopObjectDetecter = True
            elif txEmergencyData is objectDetectorStartCommand:
                stopObjectDetecter = False

def gpsDataUpdater():
    NMEA_buff =0
    lat_in_degrees =0
    long_in_degrees =0
    ser = serial.Serial("/dev/serial0")  # Open port with baud rate
    global txGPSData

    while stopGpsDataUpdater is not True:

        def GPS_Info():

            nmea_time = []
            nmea_latitude = []
            nmea_longitude = []
            nmea_time = NMEA_buff[0]  # extract time from GPGGA string
            nmea_latitude = NMEA_buff[1]  # extract latitude from GPGGA string
            nmea_longitude = NMEA_buff[3]  # extract longitude from GPGGA string

            print("NMEA Time: ", nmea_time, '\n')
            print("NMEA Latitude:", nmea_latitude, "NMEA Longitude:", nmea_longitude, '\n')

            lat = float(nmea_latitude)  # convert string into float for calculation
            longi = float(nmea_longitude)  # convertr string into float for calculation

            lat_in_degrees = convert_to_degrees(lat)  # get latitude in degree decimal format
            long_in_degrees = convert_to_degrees(longi)  # get longitude in degree decimal format

        # convert raw NMEA string into degree decimal format
        def convert_to_degrees(raw_value):
            decimal_value = raw_value / 100.00
            degrees = int(decimal_value)
            mm_mmmm = (decimal_value - int(decimal_value)) / 0.6
            position = degrees + mm_mmmm
            position = "%.4f" % (position)
            return position

        gpgga_info = "$GPGGA,"
        ser = serial.Serial("/dev/serial0")  # Open port with baud rate
        GPGGA_buffer = 0

        while True:
            received_data = (str)(ser.readline())  # read NMEA string received
            GPGGA_data_available = received_data.find(gpgga_info)  # check for NMEA GPGGA string
            if (GPGGA_data_available > 0):
                GPGGA_buffer = received_data.split("$GPGGA,", 1)[1]  # store data coming after "$GPGGA," string
                NMEA_buff = (GPGGA_buffer.split(','))  # store comma separated data in buffer
                GPS_Info()  # get time, latitude, longitude

                txGPSData = lat_in_degrees * (10 ** 4) * (10 ** 6) + long_in_degrees * (10 ** 4)
def gyroDataUpdater():
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    INT_ENABLE = 0x38
    ACCEL_XOUT_H = 0x3B
    ACCEL_YOUT_H = 0x3D
    ACCEL_ZOUT_H = 0x3F
    GYRO_XOUT_H = 0x43
    GYRO_YOUT_H = 0x45
    GYRO_ZOUT_H = 0x47
    Gx = 0
    Gy = 0
    Gz = 0
    global txGyroData

    while stopGyroDataUpdater is not True:

        def MPU_Init():
            # write to sample rate register
            bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)

            # Write to power management register
            bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)

            # Write to Configuration register
            bus.write_byte_data(Device_Address, CONFIG, 0)

            # Write to Gyro configuration register
            bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)

            # Write to interrupt enable register
            bus.write_byte_data(Device_Address, INT_ENABLE, 1)
        def read_raw_data(addr):
            # Accelero and Gyro value are 16-bit
            high = bus.read_byte_data(Device_Address, addr)
            low = bus.read_byte_data(Device_Address, addr + 1)

            # concatenate higher and lower value
            value = ((high << 8) | low)

            # to get signed value from mpu6050
            if (value > 32768):
                value = value - 65536
            return value

        bus = smbus.SMBus(1)  # or bus = smbus.SMBus(0) for older version boards
        Device_Address = 0x68  # MPU6050 device address

        MPU_Init()

        print(" Reading Data of Gyroscope and Accelerometer")

        while True:
            # Read Accelerometer raw value
            acc_x = read_raw_data(ACCEL_XOUT_H)
            acc_y = read_raw_data(ACCEL_YOUT_H)
            acc_z = read_raw_data(ACCEL_ZOUT_H)

            # Read Gyroscope raw value
            gyro_x = read_raw_data(GYRO_XOUT_H)
            gyro_y = read_raw_data(GYRO_YOUT_H)
            gyro_z = read_raw_data(GYRO_ZOUT_H)

            # Full scale range +/- 250 degree/C as per sensitivity scale factor
            Ax = acc_x / 16384.0
            Ay = acc_y / 16384.0
            Az = acc_z / 16384.0

            Gx = gyro_x / 131.0
            Gy = gyro_y / 131.0
            Gz = gyro_z / 131.0

            # print("Gx=%.2f" % Gx, u'\u00b0' + "/s", "\tGy=%.2f" % Gy, u'\u00b0' + "/s", "\tGz=%.2f" % Gz, u'\u00b0' + "/s",
            # "\tAx=%.2f g" % Ax, "\tAy=%.2f g" % Ay, "\tAz=%.2f g" % Az


            # gyroVariable = Gx1*(10**6) + Gy1*(10**3) + Gz1
            # return gyroVariable

            Sf = 2  # Range of negative values is between 200 and 400
            Gx1 = int(Gx)
            Gy1 = int(Gy)
            Gz1 = int(Gz)

            if (Gx1 < 0):
                Gx1 = abs(Gx1) + 500

            if (Gy1 < 0):
                Gy1 = abs(Gy1) + 500

            if (Gz1 < 0):
                Gz1 = abs(Gz1) + 500

            Ax1 = round(Ax, 2)
            Ay1 = round(Ay, 2)
            Az1 = round(Az, 2)
            if (Ax1 > 0):
                Ax1 = Ax1 * 100
            if (Ay1 > 0):
                Ay1 = Ay1 * 100
            if (Az1 > 0):
                Az1 = Az1 * 100
            if (Ax1 < 0):
                Ax1 = (abs(Ax1) + Sf) * 100
            if (Ay1 < 0):
                Ay1 = (abs(Ay1) + Sf) * 100
            if (Az1 < 0):
                Az1 = (abs(Az1) + Sf) * 100

            txGyroData = Gx1 * 10 ** 15 + Gy1 * 10 ** 12 + Gz1 * 10 ** 9 + int(Ax1 * (10 ** 6) + Ay1 * (10 ** 3) + Az1)

            # acceloVariable1 = Ax1*(10**3) + Ay1


            # print("\tAx=%.2f g" % Ax, "\tAy=%.2f g" % Ay, "\tAz=%.2f g" % Az,accelo1,gyro1)

            # print("Gx=%.2f" % Gx, u'\u00b0' + "/s", "\tGy=%.2f" % Gy, u'\u00b0' + "/s", "\tGz=%.2f" % Gz, u'\u00b0' + "/s",
            #     "\tAx=%.2f g" % Ax, "\tAy=%.2f g" % Ay, "\tAz=%.2f g" % Az,GyroData)
            #sleep(1)
def ultraSonicDataUpdater():
    # GPIO Mode (BOARD / BCM)
    global txUltraSonicData
    GPIO.setmode(GPIO.BCM)

    # set GPIO Pins
    GPIO_TRIGGERA = 27

    GPIO_ECHOA = 22
    GPIO_TRIGGERB = 21
    GPIO_ECHOB = 20

    while stopUltraSonicDataUpdater is not True:

        # set GPIO direction (IN / OUT)
        GPIO.setup(GPIO_TRIGGERA, GPIO.OUT)
        GPIO.setup(GPIO_ECHOA, GPIO.IN)
        GPIO.setup(GPIO_TRIGGERB, GPIO.OUT)
        GPIO.setup(GPIO_ECHOB, GPIO.IN)

        GPIO.output(GPIO_TRIGGERA, True)
        GPIO.output(GPIO_TRIGGERB, True)
        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(GPIO_TRIGGERA, False)
        GPIO.output(GPIO_TRIGGERB, False)

        StartTimeA = time.time()
        StopTimeA = time.time()
        StartTimeB = time.time()
        StopTimeB = time.time()

        # save StartTime
        while GPIO.input(GPIO_ECHOA) == 0:
            StartTimeA = time.time()
        while GPIO.input(GPIO_ECHOB) == 0:
            StartTimeB = time.time()
            # save time of arrival
        while GPIO.input(GPIO_ECHOA) == 1:
            StopTimeA = time.time()
        while GPIO.input(GPIO_ECHOB) == 1:
            StopTimeB = time.time()

            # time difference between start and arrival
        TimeElapsedA = StopTimeA - StartTimeA
        TimeElapsedB = StopTimeB - StartTimeB
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distanceA = int((TimeElapsedA * 34300) / 2)
        distanceB = int((TimeElapsedB * 34300) / 2)

        txUltraSonicData = distanceA * 1000 + distanceB
        #time.sleep(1)


if __name__=='__main__':
    print('\nStarting Threads...')
    motionDataUpdaterThread = Thread(target = motionDataUpdater)
    camDataUpdaterThread = Thread(target = camDataUpdater)
    specialDataUpdaterThread = Thread(target=specialDataUpdater)
    emergencyDataUpdaterThread = Thread(target=emergencyDataUpdater)

    motionControllerThread = Thread(target = motionController)
    camControllerThread = Thread(target = camController)
    specialControllerThread = Thread(target = specialController)
    emergencyControllerThread = Thread(target = emergencyController)

    videoStreamerThread = Thread(target = videoStreamer)
    objectDetectorThread = Thread(target=objectDetector)

    #motionDataUpdaterThread.start()
    #camDataUpdaterThread.start()
    #specialDataUpdaterThread.start()
    #emergencyDataUpdaterThread.start()
    #time.sleep(1)
    #stopEmergencyDataUpdater =True

    #motionControllerThread.start()

    #camControllerThread.start()
    #specialControllerThread.start()
    #emergencyControllerThread.start()
    videoStreamerThread.start()

    objectDetectorThread.start()


