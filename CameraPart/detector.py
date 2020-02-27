import time
import RPi.GPIO as GPIO ##General Purpose Input/Output
GPIO.setmode(GPIO.BCM)
GPIO.setup(21,GPIO.IN)
from picamera import PiCamera

camera = PiCamera()

photoNumber = 1
##this variable will be used in photo name

try:
    while True:
        time.sleep(3)
        input=GPIO.input(21) ##GPIO.input returns 1        
        if input==0:         ##when a motion detected
            print("Waiting... ")
        elif input==1:
            print("Motion detected and 3 photos were taken")            
            for i in range(3):
                camera.capture('/home/pi/Desktop/Photo/image%d.jpg' % photoNumber)
                photoNumber += 1    
finally:
        GPIO.cleanup() ##this line cleans the used ports