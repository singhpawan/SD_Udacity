'''
Author: Pawandeep Singh
Date: 11/07/2016
Project: 1
'''

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# Reading in an image
def load_image(path):
    # path = '/home/pawan/Github/CarND-LaneLines-P1/test_images/solidYellowCurve.jpg'
    image = mpimg.imread(path)
    return image


def display_image(image):
    # printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    plt.imshow(image)  # Call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    plt.show()


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_points = []
    right_points= []


    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1)/(x2 - x1)

            if abs(slope) < 0.42:
               continue

            if slope < 0:
              left_points.append((x1,y1))
              left_points.append((x2,y2))
            else:
              right_points.append((x1,y1))
              right_points.append((x2,y2))

    sorted_left = sorted(left_points, key=lambda arr: arr[1])
    sorted_right = sorted(right_points, key=lambda arr: arr[1])

    left  = np.array(sorted_left)
    right = np.array(sorted_right)

    if len(left) > 0 :
      draw_lane_lines(color, img, left, thickness)

    if len(right) > 0 :
      draw_lane_lines(color, img, right, thickness)


def draw_lane_lines(color, img, points, thickness):
    x = points[:, 0]
    y = points[:, 1]
    z = np.polyfit(x, y, 1)
    k = np.poly1d(z)
    x_updated = np.linspace(x[0], x[-1], 50, dtype=int)
    y_updated = k(x_updated).astype(int)
    lines = list(zip(x_updated, y_updated))
    for index in range(1, len(x_updated)):
        cv2.line(img, lines[index - 1], lines[index], color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Testing on all the images
def list_test_images(path):
    paths = (os.listdir(path))
    for pth in paths:
        image = load_image(path + pth)
        process_image(image)


def write_video_file():
    white_output = '/home/pawan/Github/CarND-LaneLines-P1/whitel1.mp4'
    clip1 = VideoFileClip("/home/pawan/Github/CarND-LaneLines-P1/solidYellowLeft.mp4")
    white_clip = clip1.fl_image(process_image) # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


# Testing on Videos
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    # Parameters values
    kernel_size = 5
    low_threshold = 90
    high_threshold = 180
    rho = 2               # distance resolution in pixels of the Hough grid
    theta = np.pi/180     # angular resolution in radians of the Hough grid
    threshold = 10         # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 15     # maximum gap in pixels between connectable line segments
    # Loading the image
    img_shape = image.shape
    vertices = np.array([[(0,img_shape[0]),((img_shape[1]/2)- 20, img_shape[0]*3/5 ),((img_shape[1]/2)+20,img_shape[0]*3/5),(img_shape[1],img_shape[0])]], dtype=np.int32)
    # Extract the gray scale image from the original image
    gray = grayscale(image)
    # Removing noise from the gray scale image
    blur_gray = gaussian_blur(gray, kernel_size)
    # Performing Canny edge detection
    edges = canny(blur_gray, low_threshold, high_threshold)
    # display_image(edges)
    # Extracting the region of interest based on the edges.
    masked_edges = region_of_interest(edges, vertices)
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    line_edges = weighted_img(line_image, image)
    # display_image(line_edges)
    return line_edges





def main():
    path = '/home/pawan/Github/CarND-LaneLines-P1/test_images/'
    #list_test_images(path)
    write_video_file()


# Boiler Plate code to call main
if __name__ == '__main__':
    main()


'''
Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:
cv2.inRange() for color selection
cv2.fillPoly() for regions selection
cv2.line() to draw lines on an image given endpoints
cv2.addWeighted() to coadd / overlay two images cv2.cvtColor() to grayscale or change color cv2.imwrite() to output images to file
cv2.bitwise_and() to apply a mask to an image
Check out the OpenCV documentation to learn about these and discover even more awesome functionality!
'''

