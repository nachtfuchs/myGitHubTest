# import all relevant libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle

print('Import successful.')

# Define a class to receive the characteristics of each line detection
class CLine():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients over the last n iterations
        self.coeff_history = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def Input(self, current_fit):
        # if no match was found, use the last coefficients
        if current_fit == None:
            current_fit = self.coeff_history[-1]
        else:
            self.current_fit = current_fit
            #store the coefficients
            if self.coeff_history == None: 
                # the first coefficients will be stored as current_fit, to avoid "None" in the history
                self.coeff_history = current_fit
            else:
                # accumulate the coefficients
                self.coeff_history = np.vstack((self.coeff_history, current_fit))
    
    def AveragingParameters(self, n = 3):
        # average the parameters of the last n fits
        try:
            # works only if self.coeff_history has more than one set of parameters. Set the very 
            self.best_fit = np.mean(self.coeff_history[ -n:, :], axis = 0)
        except IndexError:
            # catches the very first entry
            self.best_fit = self.current_fit
        except TypeError:
            # catches the very first entry in case of type errors
            self.best_fit = self.current_fit
        return self.best_fit
        
#####################
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    ''' @brief  This functions performs the sobel operator in x or y direction
                on the input image. It returns the gradient image for the 
                chosen direction.'''
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_uint = np.uint8(abs_sobel*255/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(sobel_uint)
    binary_output[(sobel_uint >= thresh_min) & (sobel_uint <= thresh_max)] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''  @brief This functions returns a binary image where pixels equal 1 that
                fullfill the angle thresholds. Where the angle thresholds are
                violated, the pixels equal 0.'''
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_direction)
    binary_output[ (grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def hls_select(img, s_thresh=(0, 255), l_thresh =(0, 255)):
    ''' @brief  This function performs a thresholding on the S(aturation) and L(ightness) channel
                on the RGB input image.'''
    # 1) Convert to HLS color space
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    binary_S_output = np.zeros_like(img_hls[:, :, 2])
    binary_S_output[(img_hls[:, :, 2] > s_thresh[0]) & (img_hls[:, :, 2] <= s_thresh[1])] = 1
    # 3) Apply a threshold to the L channel
    binary_L_output = np.zeros_like(img_hls[:, :, 1])
    binary_L_output[(img_hls[:, :, 1] > l_thresh[0]) & (img_hls[:, :, 1] <= l_thresh[1])] = 1
    # 4) Merge the filtered channels
    binary_output = np.zeros_like(img_hls[:, :, 2])
    binary_output[ (binary_S_output == 1) & (binary_L_output == 1)] = 1 
    # 3) Return a binary image of threshold result
    return binary_output

# Function to reject the point outliers by comparing to the median value
def RejectOutlier(x_array, y_array):
    median = np.median(x_array)
    dev = x_array - median
    ind = []
    for i, x in enumerate(dev):
        if abs(x) > 200:
            ind.append(i)
    x_array = np.delete(x_array, ind)
    y_array = np.delete(y_array, ind)
    return x_array, y_array

def fit_lane_lines(binary_warped, nwindows = 9, margin = 100, minpix = 100):
    ''' @brief Find lane lines for an input warped image.
        @input binary_warped Image with ONE color channel from bird's eye view with lane lines
        @input nwindows      Choose the number of sliding windows
        @input margin        Set the width of the windows +/- margin
        @input minpix        Set minimum number of pixels found to recenter window
        
        @output left_fit A set of polynomial parameters that describe a parabola line 
                         for the left lane line
        @output right_fit A set of polynomial parameters that describe a parabola line 
        for the right lane line
    '''
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    left_offset = int(0.2* binary_warped.shape[1]) # avoid searching from the very border of the left side. Take an offset in percent
    right_offset = int(0.9* binary_warped.shape[1])
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[left_offset:midpoint]) + left_offset
    rightx_base = np.argmax(histogram[midpoint:right_offset]) + midpoint
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    leftx, lefty = RejectOutlier(leftx, lefty)
    
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    rightx, righty = RejectOutlier(rightx, righty)
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit

def averageLaneLines(line_fit, line_object):
    ''' @brief  Average the lane line for a given object line_object of class CLine 
                and a line fit, which is a set of parameters
        
        @input line_fit A set of parameters for a line. E.g. for parabola: (1, 2, 0)
        @input line_object Object of CLine
    '''
    result = line_object.AveragingParameters(line_fit, n = 3)
    return result

def CompareParameters(current_fit, line_object, rtol = 0.1, atol = 0.5):
    ''' @brief  Compare parameters of the current set with the set from the step before.
                Accept new parameter sets only if they are within a certain boundary of
                the old parameter set.
        @input current_fit Current parameter set E.g. for parabola: (1, 2, 0)
    '''
    # Deal with the very first element
    if line_object.current_fit[0] == False:
        return current_fit
    else:
        # handle all the rest, by comparison
        arr_equal = np.allclose(current_fit, line_object.current_fit, rtol, atol)
        if arr_equal == True:
            # accept the current value
            return current_fit
        else:
            # return the old value 
            return line_object.current_fit
    
    

#####################
# load the paths to the filenames
file_list = os.listdir(r".\camera_cal")

nx = 9 # number of inside corners in x-direction
ny = 6 # number of inside corners in y-direction

# Initialize variables for object points
objp = np.zeros( (ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Initialize arrays to store object points and image points
obj_points = [] # 3D
img_points = [] # 2D

# search for chessboard corners in the images
for file in file_list[:-1]: # '-1' to exclude the Thumbs.db file
    img = cv2.imread('./camera_cal/' + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    #if found, add object points and image points
    if ret == True:
        obj_points.append(objp)
        img_points.append(corners)
        
        #draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        write_name = './camera_cal_corners_found/corners_found' + file
        cv2.imwrite(write_name, img)

# load image for reference
img = cv2.imread('.\camera_cal\\' + file_list[0])
img_size = (img.shape[1], img.shape[0]) # x, y pixels

# perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
print('Camera calbration successful.')
#--------------------

#####################
# defining perpective transformation areas
bot_width = 0.76 # percent of bottom trapizoid height 
mid_width = 0.08 # percent of middle trapizoid height
height_pct = 0.62 # percent for trapizoid height
bottom_trim = 0.935 # percent from top to bottom to avoid car hood
# source of the four points (trapizoid)
src = np.float32([[img.shape[1]* (0.5 - mid_width/2), img.shape[0] * height_pct], 
                  [img.shape[1]* (0.5 + mid_width/2), img.shape[0] * height_pct],
                  [img.shape[1]* (0.5 + bot_width/2), img.shape[0] * bottom_trim],
                  [img.shape[1]* (0.5 - bot_width/2), img.shape[0] * bottom_trim]
                 ])
offset = img_size[0] * 0.25
dst = np.float32([[offset, 0],
                  [img_size[0] - offset, 0],
                  [img_size[0] - offset, img_size[1]],
                  [offset, img_size[1]]
                 ]) # dst needs to be a rectangle -> bird's eye view
# execute the transform for each image
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src) #used to display identified road lanes in the "original image"
#--------------------

#####################
# perform windowing on the lane line for all images
window_height = 80
margin = 100 # How much to slide left and right for searching
#--------------------

#####################
img_height = img.shape[0]
ploty = np.linspace(0, img_height-1, num = img_height) #these will be used to plot

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
y_eval = np.max(ploty)
#--------------------

#####################
# Apply to video
cap = cv2.VideoCapture('project_video.mp4')

height , width , layers =  img.shape #used for creating a video
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter('project_video_identified_lanes.avi', fourcc, 25.0,(width,height))

#instantiation
left_line = CLine()
right_line = CLine()

# run the algorithm on the images
while(True):

    # Capture frame-by-frame
    ret, img = cap.read()

    #undistort the image
    img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # perform preprocessing on undistorted images
    img_preprocessed = np.zeros_like(img_undist[:, :, 0])
    # apply the preprocessing
    grad_x = abs_sobel_thresh(img_undist, orient='x', thresh_min = 25, thresh_max = 255)
    grad_y = abs_sobel_thresh(img_undist, orient='y', thresh_min = 25, thresh_max = 255)
    color_binary = hls_select(img_undist, s_thresh = (100, 255), l_thresh = (100, 255))
    # combine the binary images
    img_preprocessed[(grad_x == 1) & (grad_y == 1) | (color_binary == 1)] =255 
    
    # perform perspective transformation to bird's eye view
    img_warped = cv2.warpPerspective(img_preprocessed, M, img_size, flags = cv2.INTER_LINEAR)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # get the lane fit
    lane_fit = fit_lane_lines(img_warped, nwindows = 9, margin = 100, minpix = 50)
    
    # calculate the polynomial fit
    left_fit_raw = lane_fit[0] # left lane parameters as a parabola
    right_fit_raw = lane_fit[1] # left lane parameters as a parabola

    # compare the current fit with the previous fit
#     left_fit_comp = CompareParameters(left_fit_raw, left_line)
#     right_fit_comp = CompareParameters(right_fit_raw, right_line)    
    
    # store the either the current fit or the last accepted fit in the line object
    left_line.Input(left_fit_raw)
    right_line.Input(right_fit_raw)
    
    # Smoothen the lane line parameters using previous coefficients
    left_fit = left_line.AveragingParameters(n = 10)
    right_fit = right_line.AveragingParameters(n = 10)
    
    # calculate the x values for all y-values (-> one x for one y pixel)
    left_fit_x = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fit_x = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix,left_fit_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fit_x*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (1280, 720))
    result = cv2.addWeighted(img_undist, 1, newwarp, 0.3, 0)
    # calculate the vehicle offset from the middle
    camera_center = (left_fit_x[-1] + right_fit_x[-1]) / 2
    center_diff = (camera_center - img_warped.shape[1]/2 ) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    
    # draw curvature information in the top left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.putText(result, 'left curve:  ' + str(left_curverad) + 'm', (20,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    result = cv2.putText(result, 'right curve: ' + str(right_curverad) + 'm', (20, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    result = cv2.putText(result, 'center deviation ' + side_pos + ': ' + str(abs(center_diff)) + 'm', (20, 120), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame', result)

    # attach to video
    video.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
print('Processing finished.')
video.release()
cap.release()
cv2.destroyAllWindows()

    
