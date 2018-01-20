import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')
print('This image is: ', type(image),
        'with dimensions: ', image.shape)

ysize = image.shape[0]
xsize = image.shape[1]
print 'x: ', xsize, ', y: ', ysize

region_select = np.copy(image)

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

region_select[region_threshold] = [255, 0, 0]

plt.imshow(region_select)
plt.show()
