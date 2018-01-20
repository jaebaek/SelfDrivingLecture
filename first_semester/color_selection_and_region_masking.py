import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')
print('This image is: ', type(image),
        'with dimensions: ', image.shape)

ysize = image.shape[0]
xsize = image.shape[1]
print 'x: ', xsize, ', y: ', ysize

new_image = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:,:,0] < rgb_threshold[0]) \
        | (image[:,:,1] < rgb_threshold[1]) \
        | (image[:,:,2] < rgb_threshold[2])

print('This thresholds is: ', type(thresholds),
        'with dimensions: ', thresholds.shape)

left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
print fit_left # this is [A, B] of line (y=Ax+B)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# np.meshgrid(np.arange(0, 3), np.arange(0, 5))
# returns
# [0, 1, 2] X 5,
# 3 X [0, 1, 2, 3, 4]^T
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_threshold = (YY > (XX * fit_left[0] + fit_left[1])) & \
        (YY > (XX * fit_right[0] + fit_right[1])) & \
        (YY < (XX * fit_bottom[0] + fit_bottom[1]))

# only elements of thresholds[ .. ] = False (i.e., close to white)
# and inside the region will be red-colored
new_image[~thresholds & region_threshold] = [255, 0, 0]

plt.imshow(new_image)
plt.show()
