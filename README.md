## Advanced Lane Finding Documentation
#### By: Igor Gall

****
The most rubric points to pass the project are discussed in the the jupyter notebook called 
"Advanced Lane Lines.ipynb" that is available in this repository. There I go step by step
through the rubric points and discuss each of the steps to successfully find the lane lines.

In order to execute the jupyter notebook cells successfully, the data from the repository
https://github.com/udacity/CarND-Advanced-Lane-Lines needs to be in the same directory as
the jupyter notebook. Once the jupyter notebook has access to the image files, it should
run successfully each cell.

This GitHub repository is also used to provide access to the video file project_video_identified_lanes.avi
where lane lines are identified by my image processing pipeline. It is the output video that is required for the project to pass.

The source.py file performs a camera calibration using the images from the mentioned udacity repository. Furthermore, it contains the lane line processing steps from the jupyter notebook but it was structured to be used within a script instead of a notebook. For example, there are no plots for the different steps available. The only thing that is displayed is the final image with the estimated lane lines.
