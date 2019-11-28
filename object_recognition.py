#!/usr/bin/env python
from __future__ import division

import tensorflow as tf
from tensorflow import keras
import h5py
import os
from tensorflow.python.keras.backend import set_session
import string

import cv2
import numpy as np
import time

import roslib
import sys
import rospy
from std_msgs.msg import String
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

# Define colour thresholds for masking

# Consider searching for brown to find pedestrian

# Red for crosswalks
lower_brown = np.array([10,10,20])
upper_brown = np.array([20,255,200])

lower_red = np.array([0,100,100])
upper_red = np.array([10,255,255])

# May not need
lower_white = np.array([0,0,95])
upper_white = np.array([200,3,210])

# Moving car. This one is going to be hard. Tune later.
lower_grey = np.array([0,150,50])
upper_grey = np.array([255,255,180])

# Parked car and license plate characters.
# Tuned!!
lower_black = np.array([0, 0, 0])
upper_black = np.array([255, 255, 30])

lower_blue = np.array([100, 100, 50])
upper_blue = np.array([130, 255, 255])

# Set constants describing relevant image region for each mask, and the number of
# pixels required for us to conclude the sought object is present.

# TODO: Determine appropriate values for all class variables below

# Pixel coordinates corresponding to bottom left corner of red search region in (x, y)
red_bottom_left = [280, 720]
# Pixel coordinates corresponding to top right corner of red search region in (x, y)
red_top_right = [1000, 700]

# Search for pedestrian in one range
pedestrian_bottom_left = [0, 0]
pedestrian_top_right = [0, 0]

grey_bottom_left = [0, 0]
grey_top_right = [0, 0]

blue_bottom_left = [150, 600]
blue_top_right = [350, 350]

license_bottom_left = [50, 600]
license_top_right = [450, 300]

black_bottom_left = [200, 500]
black_top_right = [350, 400]

# Number of high pixels within search region required for us to conclude that the sought object is present.
red_threshold = 500000
pedestrian_threshold = 70000
lower_license_threshold = 2500000
upper_license_threshold = 8000000
# Grey value set like this for now to prevent triggering
grey_threshold = 2000000
blue_threshold = 1000

# Image Dimensions (x, y)
IMAGE_DIMENSIONS = [1000, 1000]

# Constants for license plate croppping
Y_OFFSET = 5
FIRST_OFFSET = -203
SECOND_OFFSET = -103
THIRD_OFFSET = 98
FOURTH_OFFSET = 197

prev_mask_assigned = False
previous_brown_mask = None

sess = tf.Session()
graph = tf.get_default_graph()

#Load neural network for license plate recognition
set_session(sess)
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), "license_recognition_16.0.h5"))

nums = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']
class_names = nums + list(string.ascii_uppercase)

stop = False
starting = False

class image_processor:

    def __init__(self, teamID, password): 

        # Set publishing channels
        self.license_pub = rospy.Publisher('/license_plate', String, queue_size=10)
        self.driver_pub = rospy.Publisher('/drive', String, queue_size=10)

        # Set inputs and subscriptions
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)        #print(license_count)
        #print("-----")
        self.teamID = teamID
        self.password = password


    def callback(self, data):
        global prev_mask_assigned
        global previous_brown_mask
        global stop, starting

        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.imshow("image", image)
        # cv2.imshow("cropped image", image[300:600, 50:450])
        # cv2.waitKey(2)

        # TODO: figure out license plate ID is and how to get it.
        license_plate_id = str(0)

        # Generate array of masks for each desired colour
        brown_mask, white_mask, red_mask, blue_mask, black_mask = get_masks(image)

        # Initial condition for previous_white_mask
        if (prev_mask_assigned == False): 
            previous_brown_mask = brown_mask
            prev_mask_assigned = True

        
        # Search for license plates and crosswalk in pre-defined region.
        license_count = np.sum(blue_mask[license_top_right[1]:license_bottom_left[1], license_bottom_left[0]:license_top_right[0]])
        red_count = np.sum(red_mask[red_top_right[1]:red_bottom_left[1], red_bottom_left[0]:red_top_right[0]])

        # If you are at a crosswalk, stop!
        if (red_count > red_threshold):
            stop = True

        # If stopped, look for pedestrian
        if stop == True:
            subtracted_count = np.sum(brown_mask - previous_brown_mask)
            if subtracted_count < pedestrian_threshold:
                stop = False
                starting = True

        # Map Boolean to corresponding String for publishing
        if stop == True: 
		drive_to_publish = "Stop"

        else: 
		drive_to_publish = "Go"



        # Publish stop boolean to drive node
        self.driver_pub.publish(drive_to_publish)

        # If just starting up from crosswalk, sleep to prevent double counting of crosswalk
        if starting == True:
            starting = False
            time.sleep(5)

        if(license_count > lower_license_threshold and license_count < upper_license_threshold):
            delta, pts = search_image(white_mask, license_bottom_left, license_top_right)

            projection = four_point_transform(image, np.asarray(pts))
            #cv2.imshow("Projection", projection)
            j = 0
            
            cropped = []

            while j < 4:
                gray_char = cv2.cvtColor(crop(projection, j), cv2.COLOR_BGR2GRAY) / 255.0
                resized = cv2.resize(gray_char, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                cropped.append(resized)
                #cv2.imshow(str(j), cropped[j])
                j += 1

            #cv2.imshow("Counted_black", image[pts[0][1] - int(delta / 2):pts[0][1], pts[0][0] + int((-1*pts[0][0] + pts[3][0]) / 2):pts[3][0]])
            parking_image = image[pts[0][1] - int(delta / 2.4):pts[0][1] - 5, pts[0][0] + int((-1*pts[0][0] + pts[3][0]) / 2):pts[3][0]]
            resized_parking = cv2.resize(parking_image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            gray_parking = cv2.cvtColor(resized_parking, cv2.COLOR_BGR2GRAY) / 255.0
            blurred_gray_parking = cv2.blur(gray_parking, (9,9))
            # _,  binary_parking = cv2.threshold(gray_parking, 50, 1, cv2.THRESH_BINARY)


            cropped.append(blurred_gray_parking)

            #cv2.imshow("Parking Number", blurred_gray_parking)
            # cv2.imshow("Black Parking", black_mask[pts[0][1] - int(delta / 2.4):pts[0][1] - 5, pts[0][0] + int((-1*pts[0][0] + pts[3][0]) / 2):pts[3][0]])
            # cv2.imshow("White Parking", white_mask[pts[0][1] - int(delta / 2.4):pts[0][1] - 5, pts[0][0] + int((-1*pts[0][0] + pts[3][0]) / 2):pts[3][0]])

            #time.sleep(2)

            #black_count = np.sum(black_mask[pts[0][1] - int(delta / 2.4):pts[0][1] - 5, pts[0][0] + int((-1*pts[0][0] + pts[3][0]) / 2):pts[3][0]])
            #white_count = np.sum(white_mask[pts[0][1] - int(delta / 2.4):pts[0][1] - 5, pts[0][0] + int((-1*pts[0][0] + pts[3][0]) / 2):pts[3][0]])



            license_characters = process_license(np.asarray(cropped))


            # #Format license characters and publish to ROS Master
            # #Example: 'TeamRed,multi21,4,XR58'

            plate_to_publish = self.teamID + ',' + self.password + ',' + license_characters
            print(plate_to_publish)
            
            self.license_pub.publish(plate_to_publish) 

            # centre1 = pts[0][0], pts[0][1]
            # centre2 = pts[1][0], pts[1][1]
            # centre3 = pts[2][0], pts[2][1]
            # centre4 = pts[3][0], pts[3][1]

            # cv2.circle(image, centre1, 5, (0, 20, 200), 2)
            # cv2.circle(image, centre2, 5, (0, 20, 200), 2)
            # cv2.circle(image, centre3, 5, (0, 20, 200), 2)
            # cv2.circle(image, centre4, 5, (0, 20, 200), 2)
            #cv2.imshow("Projection", projection)
            #time.sleep(2)

        #cv2.imshow("image", image)
        #cv2.imshow("Subtracted Mask", brown_mask - previous_brown_mask)
        cv2.waitKey(2)

        previous_brown_mask = brown_mask

        # # Initial condition for previous_white_mask
        # if (prev_mask_assigned == False): 
        #     previous_white_mask = white_mask
        #     prev_mask_assigned = True

        # # Now need to determine environment state and provide instructrions to drive system

        # # Determine response to enviroment
        # shouldStop = determine_robot_response(red_mask, grey_mask, white_mask)

        # if (shouldStop == True): drive_to_publish = "Stop"
        # else: drive_to_publish = "Go"

        # # Publish shouldStop boolean to drive node
        # #self.driver_pub.publish(drive_to_publish)

        # # Now look for license
        # #found_plate, transformed_image = find_license(white_mask, blue_mask, image)
        
        # # If license is found, detect characters
        # if found_plate == True: 
        #     license_characters = process_license(transformed_image)

        #     # Format license characters and publish to ROS Master
        #     # Example: 'TeamRed,multi21,4,XR58'
        #     plate_to_publish = self.teamID + ',' + self.password + ',' + license_plate_id + ',' + license_characters
        #     self.license_pub.publish(plate_to_publish) 

        # # Update white mask for next frame.
        # previous_white_mask = white_mask

# Filters image for red, white, grey, and blue.
# Parameter: image - image you wish to mask. Pass in BGR format
# Returns: An array of images in OpenCV binary format, each corresponding to masks filtering for a different colour.
#          Returned array is in the form [red_mask, white_mask, grey_mask, blue_mask]
def get_masks(image):

    global lower_brown, upper_brown, lower_white, upper_white, lower_grey, upper_grey, lower_blue, upper_blue

    # Convert image to HSV format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Generate masks using defined thresholds
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # # Check masks (Debugging step)
    #cv2.imshow("black_mask", black_mask[black_top_right[1]:black_bottom_left[1], black_bottom_left[0]:black_top_right[0]])
    #cv2.imshow("white_mask", white_mask[blue_top_right[1]:blue_bottom_left[1], blue_bottom_left[0]:blue_top_right[0]])
    # cv2.imshow("grey_mask", grey_mask)
    #cv2.imshow("blue_mask", blue_mask[blue_top_right[1]:blue_bottom_left[1], blue_bottom_left[0]:blue_top_right[0]])
    #cv2.imshow("white_mask", white_mask[300:600, 50:450])
    #cv2.imshow("Red Mask", red_mask)
    #cv2.imshow("Bounded Red Mask", red_mask[red_top_right[1]:red_bottom_left[1], red_bottom_left[0]:red_top_right[0]])
    #cv2.imshow("Brown Mask", brown_mask)
    cv2.waitKey(2)

    # Return masks 
    return brown_mask, white_mask, red_mask, blue_mask, black_mask
    
# Iterates through region of a binary mask image, counting the number of high pixels.
# Parameter: mask - 2D numpy array - a binary mask of image that you wish to search
# Parameter: bottom_left - int - (x, y) coordinates corresponding to the bottom
#            left corner of the desired rectangular search region
# Parameter: top_right - int - Same as above, but for top right corner
# Returns: pixel_count - int - number of pixels detected
# Returns: pts - 2D array of ints - array of 4 (x,y) pts corresponding to the extrema of the detected image.
def search_image(mask, bottom_left, top_right):
    # Declare variables to store extrema
    smallest_x = IMAGE_DIMENSIONS[0]
    y_of_smallest_x = 0

    pixel_count = 0

    saved_smallest_x = 0
    saved_y_of_smallest_x = 0

    nonzero_streak = 0
    
    step = 1
    y = top_right[1]
    x = bottom_left[0] + int((top_right[0] - bottom_left[0]) / 6.0)

    num_iterations = 0

    # Iterate through all pixels in region, searching for those that passed colour filtering
    while x < top_right[0]:
        y = top_right[1]
        while y < bottom_left[1]:

            if mask[y, x] !=  0:
                # Only looking for left hand corner
                if smallest_x != IMAGE_DIMENSIONS[0]:
                    break

                nonzero_streak += 1

                # If x is the smallest we've found, and we're not currently screening a candidate for smallest x
                if x < smallest_x and saved_smallest_x == 0:
                    saved_y_of_smallest_x, saved_smallest_x = y, x
                    nonzero_streak = 0

                if nonzero_streak == 10:

                    if saved_smallest_x != 0: 
                        y_of_smallest_x, smallest_x = saved_y_of_smallest_x, saved_smallest_x
                        saved_smallest_x = 0
                    
                    nonzero_streak = 0

                # Iterate pixel count
                pixel_count += 1
            else:
                # Streak over. Reset values.
                nonzero_streak = 0
                saved_smallest_x = 0
            y += step
        x += step

    #return pixel_count, [smallest_x, y_of_smallest_x]
    
    x = smallest_x
    y = y_of_smallest_x

    zero_streak = 0

    # Finds top right corner of large white block

    delta = 0

    while zero_streak < 10:
        if x >= mask.shape[1]:
            break

        if mask[y, x] == 0:
            zero_streak += 1
        else:
            zero_streak = 0
        x += 1
        delta += 1
    
    block_top_right_x = x - 10
    block_top_right_y = y

    x = smallest_x
    y = y_of_smallest_x

    zero_streak = 0

    # Find top righthand corner of license

    while zero_streak < 10:
        if y >= mask.shape[0]:
            break

        if mask[y, x + 5] == 0:
            zero_streak += 1
        else:
            zero_streak = 0
        y += 1

    corner1_x = x
    corner1_y = y - 10
    nonzero_streak = 0

    # Find bottom righthand corner of license

    while nonzero_streak < 10:
        if y > 719:
            break
        if mask[y, x] != 0:
            nonzero_streak += 1
        else:
            nonzero_streak = 0
        y += 1
    
    corner2_x = x
    corner2_y = y - 10
    
    x = block_top_right_x - 5
    y = block_top_right_y
    zero_streak = 0

    while zero_streak < 10:
        if mask[y, x] == 0:
            zero_streak += 1
        else:
            zero_streak = 0
        y += 1
        if y == IMAGE_DIMENSIONS[1]:
            break
    
    corner3_x = block_top_right_x
    corner3_y = y - 10

    nonzero_streak = 0

    while nonzero_streak < 10:
        if y > 719:
            break
        if mask[y, x] != 0:
            nonzero_streak += 1
        else:
            nonzero_streak = 0
        y += 1
    
    corner4_x = x + 5
    corner4_y = y - 10
    # Generate array of points describing extrema of license plate.
    #pts = [[smallest_x, y_of_smallest_x], [largest_x, y_of_largest_x], [x_of_smallest_y, smallest_y], [x_of_largest_y, largest_y]]

    # Return corners of white region
    pts = [[corner1_x, corner1_y], [corner2_x, corner2_y], [corner3_x, corner3_y], [corner4_x, corner4_y]]

    return delta, pts

# Filters through masks to search for different obstacles.
# Sets booleans describing their presence.
def get_environment_state(red_mask, grey_mask, white_mask):
    global previous_white_mask
    global red_bottom_left, red_top_right, grey_bottom_left, grey_top_right, pedestrian_bottom_left, pedestrian_top_right
    global red_threshold, grey_threshold
    
    # Generate image with non-zero values only for pixels that changed from previous frame
    delta_white_mask = np.subtract(white_mask, previous_white_mask)

    # Get number of high pixels from each mask
    crosswalk_pixel_count, _ = search_image(red_mask, red_bottom_left, red_top_right)
    car_pixel_count, _ = search_image(grey_mask, grey_bottom_left, grey_top_right)
    pedestrian_pixel_count, _ = search_image(delta_white_mask, pedestrian_bottom_left, pedestrian_top_right)

    # Compare number of pixels with pre-determined thresholds to set booleans describing presence of sought object.
    crosswalk_in_image = crosswalk_pixel_count >= red_threshold
    car_in_image = car_pixel_count >= grey_threshold

    # If some pixels are non-zero, pedestrian has moved. If all are zero, pedestrian has stopped, and our robot is good to drive.
    pedestrian_in_crosswalk =  pedestrian_pixel_count != pedestrian_threshold

    return crosswalk_in_image, car_in_image, pedestrian_in_crosswalk

# Recognize and return the license plate characters once license plate flagged in an image.
def process_license(character_images):

    global graph

    # Now make the predictions using the pre-trained model
    
    character_data = np.expand_dims(character_images, 3)

    with graph.as_default():
        set_session(sess)
        predictions = model.predict(character_data)
    

    i = 0
    plate_as_chars = []
    while i < 5:
        plate_as_chars.append(class_names[np.argmax(predictions[i])])
        i += 1
    
    # # Correct potential bad values
    # No Ts in second half of license plate. These are a 7.
    if plate_as_chars[2] == 'T':
        plate_as_chars[2] = '7'
        print("Corrected T -> 7 at i=2")

    if plate_as_chars[3] == 'T':
        plate_as_chars[3] = '7'
        print("Corrected T -> 7 at i=3")

    if plate_as_chars[0] == '2':
        plate_as_chars[0] = 'Z'
        print("Corrected 2 -> Z at i=0")

    if plate_as_chars[1] == '2':
        plate_as_chars[1] = 'Z'
        print("Corrected 2 -> Z at i=1")

    if plate_as_chars[2] == 'H':
        plate_as_chars[2] = '8'
        print("Corrected H -> 8 at i=2")

    if plate_as_chars[3] == 'H':
        plate_as_chars[3] = '8'
        print("Corrected H -> 8 at i=3")

    if plate_as_chars[2] == 'Z':
        plate_as_chars[2] = '2'
        print("Corrected Z -> 2 at i=2")

    if plate_as_chars[3] == 'Z':
        plate_as_chars[3] = '2'
        print("Corrected Z -> 2 at i=3")
    
    if plate_as_chars[2] == 'I':
        plate_as_chars[2] = '1'
        print("Corrected I -> 1 at i=2")

    if plate_as_chars[3] == 'I':
        plate_as_chars[3] = '1'
        print("Corrected I -> 1 at i=3")
    
    if plate_as_chars[0] == '1':
        plate_as_chars[0] = 'I'
        print("Corrected 1 -> I at i=0")

    if plate_as_chars[1] == '1':
        plate_as_chars[1] = 'I'
        print("Corrected 1 -> I at i=1")

    if plate_as_chars[-1] == '0':
        plate_as_chars.append('6')
        print("Corrected 0 -> 6")

    elif plate_as_chars[-1] == 'T':
        plate_as_chars.append('1')
        print('Made correction T -> 1')

    elif plate_as_chars[-1] == 'Z':
        plate_as_chars[-1].append('2')
        print('Made correction Z -> 2')

    # Return as single string describing license plate
    return ''.join(plate_as_chars[0:4]) + ',' + plate_as_chars[-1]
        
# Orders points so that they can be interpreted by our image transform function.
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

# Search for license plate in frame. Returns a boolean stating whether license plate was found,
# and, if so, a normal projection of that license plate. If not, just returns the passed image.
# Parameter: white_mask - 2D numpy array - white mask for filtered white of license plate
# Parameter: blue_mask - 2D numpy array - blue mask filtered for blue of parked car
# Parameter: image - 2D numpy array - original image associated with mask
# Returns: boolean indicating whether plate was found
# Returns: if plate found, a normal projection of license plate. If not, returns original image
def find_license(white_mask, blue_mask, image):
    global blue_bottom_left, blue_top_right, license_bottom_left, license_top_right
    transformed_image = image

    parked_car_pixel_count, _ = search_image(blue_mask, blue_bottom_left, blue_top_right)
    license_pixel_count, pts = search_image(white_mask, license_bottom_left, license_top_right)

    if license_pixel_count >= license_threshold:
        found_plate = True
        transformed_image = four_point_transform(image, np.asarray(pts))
    else : found_plate = False

    return found_plate, transformed_image

    
# Given masks of current frame, returns a boolean indicating whether robot should stop.
def determine_robot_response(red_mask, grey_mask, white_mask):
    stop = False

    # Set booleans describing presence of obstacles in current frame.
    crosswalk_found, car_found, pedestrian_found = get_environment_state(red_mask, grey_mask, white_mask)
    
    # If crosswalk in frame, stop. Start when pedestrian is not crossing.
    if crosswalk_found == True and pedestrian_found == True: Stop = True

    # If moving car in frame, stop.
    if car_found == True: stop = True

    return stop

# Crops license plate to extract individaul characters, dependent on number of desired character
# Leftmost character is the 0th.
def crop(img, charNum):
    # Scaling offsets, as they must accept inputs of diverse dimensions
    cropx = int(img.shape[1] * 97/600.0)
    cropy = int(img.shape[0] * 1.1)

    yOffset = Y_OFFSET * img.shape[1] / 298.0

    if (charNum == 0): xOffset = int(FIRST_OFFSET * img.shape[1] / 600.0)
    if (charNum == 1): xOffset = int(SECOND_OFFSET * img.shape[1] / 600.0)
    if (charNum == 2): xOffset = int(THIRD_OFFSET * img.shape[1] / 600.0)
    if (charNum == 3): xOffset = int(FOURTH_OFFSET * img.shape[1] / 600.0)

    y,x,_ = img.shape
    startx = int(x//2-(cropx//2) + xOffset)
    starty = int(y//2-(cropy//2) + yOffset)

    return img[starty:starty+cropy,startx:startx+cropx]

# Given a normal projection of the license plate, returns an array of images of its 
# individual characters
def get_license_character_images(image):
    character_images = []

    # Arbritary index
    i = 0
    # Loop through all 4 characters in license plate
    while i < 4:
        character_images.append(crop(image, i))
    
    return character_images

def main(args):
    rospy.init_node('image_processor', anonymous=True)
    ic = image_processor('teamAwesome', 'p@55w04d')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
