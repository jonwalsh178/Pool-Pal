"""
Motor controls for Pool-Pal functions
Authors: Jonathan Walsh, Haixin Zhao. For any questions contact: jonwalsh@udel.edu, zhx@udel.edu

"""
import pygame
import RPi.GPIO as GPIO
import time
import test_inference

# Pin setup
in1, in2, ena = 24, 23, 25
in3, in4, enb = 5, 6, 26

GPIO.setmode(GPIO.BCM)
GPIO.setup([in1, in2, in3, in4], GPIO.OUT)
GPIO.setup([ena, enb], GPIO.OUT)


GPIO.output([in1, in2, in3, in4], GPIO.LOW)

pR = GPIO.PWM(ena, 255)
pL = GPIO.PWM(enb, 255)
pR.start(0)
pL.start(0)

pygame.init()
joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
for joystick in joysticks:
    joystick.init()

def handle_motors(x, y):
    if y < -0.9:  # Forward
        pR.ChangeDutyCycle(100)
        pL.ChangeDutyCycle(100)
        GPIO.output([in1, in3], GPIO.HIGH)
        GPIO.output([in2, in4], GPIO.LOW)
        print("Moving Forward-4")
#    elif -0.9<y<-0.6:  # Forward
#        pR.ChangeDutyCycle(75)
#        pL.ChangeDutyCycle(75)
#       GPIO.output([in1, in3], GPIO.HIGH)
#        GPIO.output([in2, in4], GPIO.LOW)
#        print("Moving Forward-3")   
#    elif -0.6< y < -0.3:  # Forward
#        pR.ChangeDutyCycle(50)
#        pL.ChangeDutyCycle(50)
#        GPIO.output([in1, in3], GPIO.HIGH)
#        GPIO.output([in2, in4], GPIO.LOW)
#        print("Moving Forward-2")
#    elif -0.3< y < 0:  # Forward
#        pR.ChangeDutyCycle(25)
#        pL.ChangeDutyCycle(25)
#        GPIO.output([in1, in3], GPIO.HIGH)
#        GPIO.output([in2, in4], GPIO.LOW)
#        print("Moving Forward-1")

    elif y > 0.9:  # Backward
        pR.ChangeDutyCycle(100)
        pL.ChangeDutyCycle(100)
        GPIO.output([in2, in4], GPIO.HIGH)
        GPIO.output([in1, in3], GPIO.LOW)
        print("Moving Backward-4")
#    elif 0.6<y<0.9:  # Backward
#        pR.ChangeDutyCycle(75)
#        pL.ChangeDutyCycle(75)
#        GPIO.output([in2, in4], GPIO.HIGH)
#        GPIO.output([in1, in3], GPIO.LOW)
#        print("Moving Backward-3")   
#    elif 0.3< y < 0.6:  # Backward
#        pR.ChangeDutyCycle(50)
#        pL.ChangeDutyCycle(50)
#        GPIO.output([in2, in4], GPIO.HIGH)
#        GPIO.output([in1, in3], GPIO.LOW)
#        print("Moving Backward-2")
#    elif 0.1< y < 0.3:  # Backward
#        pR.ChangeDutyCycle(25)
#        pL.ChangeDutyCycle(25)
#        GPIO.output([in2, in4], GPIO.HIGH)
#        GPIO.output([in1, in3], GPIO.LOW)
#        print("Moving Backward-1")

    elif x < -0.9:  # Left
        pR.ChangeDutyCycle(100)
        pL.ChangeDutyCycle(100)
        GPIO.output([in1, in4], GPIO.HIGH)
        GPIO.output([in2, in3], GPIO.LOW)
        print("Turning Left-4")
#    elif -0.9<x<-0.6 :  # Left
#        pR.ChangeDutyCycle(100)
#        pL.ChangeDutyCycle(50)
#        GPIO.output([in1, in4], GPIO.HIGH)
#        GPIO.output([in2, in3], GPIO.LOW)
#        print("Turning Left-3")   
#    elif -0.6 <x< -0.3 :  # Left
#        pR.ChangeDutyCycle(75)
#        pL.ChangeDutyCycle(37)
#        GPIO.output([in1, in4], GPIO.HIGH)
#        GPIO.output([in2, in3], GPIO.LOW)
#        print("Turning Left-2")
#    elif 0.1 <x< 0.3 :  # Left
#        pR.ChangeDutyCycle(50)
#        pL.ChangeDutyCycle(25)
#        GPIO.output([in1, in4], GPIO.HIGH)
#        GPIO.output([in2, in3], GPIO.LOW)
#        print("Turning Left-1")

    elif x > 0.9:  # Right
        pR.ChangeDutyCycle(100)
        pL.ChangeDutyCycle(100)
        GPIO.output([in2, in3], GPIO.HIGH)
        GPIO.output([in1, in4], GPIO.LOW)
        print("Turning Right-4")
#    elif 0.6 <x< 0.9 :  # Left
#        pR.ChangeDutyCycle(50)
#        pL.ChangeDutyCycle(100)
#        GPIO.output([in2, in3], GPIO.HIGH)
#        GPIO.output([in1, in4], GPIO.LOW)
#        print("Turning Right-3")   
#    elif 0.3 <x< 0.6 :  # Left
#        pR.ChangeDutyCycle(37)
#        pL.ChangeDutyCycle(75)
#        GPIO.output([in2, in3], GPIO.HIGH)
#        GPIO.output([in1, in4], GPIO.LOW)
#        print("Turning Right-2")
#    elif 0.1 <x< 0.3 :  # Left
#        pR.ChangeDutyCycle(25)
#        pL.ChangeDutyCycle(50)
#        GPIO.output([in2, in3], GPIO.HIGH)
#        GPIO.output([in1, in4], GPIO.LOW)
#        print("Turning Right-1")


    else:
        pR.ChangeDutyCycle(0)
        pL.ChangeDutyCycle(0)
        GPIO.output([in1, in2, in3, in4], GPIO.LOW)
        print("Stop")

def controller():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([in1, in2, in3, in4], GPIO.OUT)
    GPIO.setup([ena, enb], GPIO.OUT)
    GPIO.output([in1, in2, in3, in4], GPIO.LOW)
    controlling = True
    print("Press Y to go back (press A or B to enter a mode after Y).")
    while controlling:      
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                # Use the joystick list to access the joystick object
                joystick = joysticks[event.joy]
                x_axis = joystick.get_axis(0)
                y_axis = joystick.get_axis(1)
                handle_motors(x_axis, y_axis)
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 4:
                    print(event)
                    controlling = False
                    pR.stop()
                    pL.stop()



#GPIO.cleanup()  # Clean up GPIO to ensure all pins are reset


	
