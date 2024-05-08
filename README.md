# Pool-Pal
Senior Design project - contains code for controlling a RC boat as well as an autonomous mode that utilizes object detection to detect debris in a pool.

Controller based pool skimmer: Connect a wireless bluetooth Xbox controller to the internal RPi. The left joystick will spin the props to move the boat. (Button B)

Autonomous based pool skimmer: Utilizes two sets of weights from training (one for detecting waste and another for things that need to be avoided) to place bounding boxes. From these bounding boxes, the algorithm will decide whether the object needs to be collected or avoided. Objects of interest will be centered in the camera's view (closer objects first) so that the boat can go forward towards the object. (Button A)

Warnings: DO NOT OBSTRUCT THE PROPELLERS OR CAMERA VIEW. AVOID SUBMERGING BOAT IN WATER.
