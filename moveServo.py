import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(03, GPIO.OUT)

pwm=GPIO.PWM(03, 50)
pwm.start(0)

def SetAngle(angle):
	duty = angle / 18. + 2.
	print duty
	GPIO.output(03, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
#	GPIO.output(03, False)
#	pwm.ChangeDutyCycle(0)

angle = 90
dir = 10
while 1:
	angle += dir
	SetAngle(angle)
	if angle+dir > 180 or angle+dir < 0:
		dir *= -1
	print angle

pwm.stop()
GPIO.cleanup()
