from typing import List
import math
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from PIL import Image
import cv2
def load_im(filename) -> np.ndarray:
    img = Image.open(filename)

    return np.array(img.convert("L"), dtype=np.uint8)

def crd_cart_to_homog(x: np.ndarray) -> np.ndarray:
    return np.hstack((x, 1 if x.ndim == 1 else np.ones((x.shape[0], 1))))

def crd_homog_to_cart(x: np.ndarray) -> np.ndarray:
    return (x.T[:-1] / x.T[-1]).T

def crd_map(H: np.ndarray, x: np.ndarray):
    return crd_homog_to_cart(H @ crd_cart_to_homog(x))

def crds_to_hmat(xvec: np.ndarray, xvec_hat: np.ndarray) -> np.ndarray:
    x_hat, y_hat = xvec_hat

    xvec_h = crd_cart_to_homog(xvec)
    zero = np.zeros((xvec_h.shape[0]))

    return np.block([
        [zero, -xvec_h, y_hat*xvec_h],
        [xvec_h, zero, -x_hat*xvec_h],
        [-y_hat*xvec_h, x_hat*xvec_h, zero]
    ])

def hmat_solve_h(hmats: List[np.ndarray]) -> np.ndarray:
    A = np.vstack(hmats)
    
    # compute SVD of A
    _, _, Vh = np.linalg.svd(A)

    # H matrix - eigenvector corresponding to smallest (last) eigenvalue
    H = Vh[-1].reshape((3, 3))

    return H

def warp_canvas_size(box_coords: List[np.ndarray], transform_H: np.ndarray):
    corners = transform_H(np.array(box_coords))

    return np.ceil(np.min(corners, axis=0)), np.ceil(np.max(corners, axis=0))


def warp(im: np.ndarray, box_coords: List[np.ndarray], H: np.ndarray):
    transform_H = ski.transform.ProjectiveTransform(H)
    
    off_min, off_max = warp_canvas_size(box_coords, transform_H)

    transform_S = ski.transform.SimilarityTransform(translation=-off_min)
    transform_inv = (transform_H + transform_S).inverse

    result_shape = off_max - off_min

    im_warp = ski.transform.warp(im, transform_inv, preserve_range=True, output_shape=result_shape)

    return im_warp

def topdown(im, params, box_dim=400, should_warp=True):
    im_h, im_w = im.shape

    vcrop_upper, vcrop_lower, hcrop_shift, hcrop_squeeze,  hcrop_lower = params

  #  hcrop_squeeze = 7.6
  #  hcrop_lower = 0.05

    hcrop_upper = hcrop_lower * hcrop_squeeze

    # in x, y
    box_coords = np.floor(np.array([
        [  (hcrop_upper+hcrop_shift)*im_w,   (vcrop_upper)*im_h],
        [(1-hcrop_upper+hcrop_shift)*im_w,   (vcrop_upper)*im_h],
        [            (1-hcrop_lower)*im_w, (1-vcrop_lower)*im_h],
        [              (hcrop_lower)*im_w, (1-vcrop_lower)*im_h]
    ])).astype(np.int64)

    box_map = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1]
    ]) * box_dim

    H = hmat_solve_h([crds_to_hmat(*pair) for pair in zip(box_coords, box_map)])

    if not should_warp:
        im[ski.draw.line(*box_coords[0], *box_coords[1])[::-1]] = 255
        im[ski.draw.line(*box_coords[1], *box_coords[2])[::-1]] = 255
        im[ski.draw.line(*box_coords[2], *box_coords[3])[::-1]] = 255
        im[ski.draw.line(*box_coords[3], *box_coords[0])[::-1]] = 255
    else:
        im = warp(im, box_coords, H)

    return H, im


def hough_lines(img, threshold):
    print(np.unique(img))
    r_max = math.ceil(np.sqrt(img.shape[0]**2+img.shape[1]**2))
    accumulator = np.zeros(shape=(180, r_max))
    angles = np.arange(180)*np.pi/180
    sines = np.sin(angles)
    cosines = np.cos(angles)
    angle_inds = np.arange(180)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y,x] > 0:
                rs = np.round(x*cosines + y*sines).astype(int)
                # print(rs.shape, sines.shape, rs.dtype)
                # print(rs)
                accumulator[angle_inds, rs]+=1
   # print(accumulator.mean())
    print(accumulator)
    #thetas, rs = np.nonzero(np.where(accumulator >  threshold))
#    accumulator = accumulator[:,3:]
    thetas, rs = np.nonzero(np.where(accumulator >  0.5*np.max(accumulator)))
    return angles[thetas], np.arange(r_max)[rs]

# in source data, lane width -> 3.75m, marker length -> 3m



# driver161

#file = "CULane/driver_161_90frame/06030828_0758.MP4/00630.jpg"
# params = (0.7, 0.1, -0.035)

#driver 182
# file = "CULane/driver_182_30frame/06011750_0258.MP4/02040.jpg"
file = "CULane/driver_182_30frame/06011702_0242.MP4/02040.jpg"
# file = "CULane/driver_182_30frame/06011844_0276.MP4/00030.jpg"
# #file = "CULane/driver_182_30frame/05312327_0001.MP4/00000.jpg"
params = (0.51,0.32, -0.04, 7.4, 0.05)

#driver 23
#file = "CULane/driver_23_30frame/05170623_0673.MP4/00020.jpg"
#file = "CULane/driver_23_30frame/05170732_0696.MP4/00200.jpg"
#file = "CULane/driver_23_30frame/05160923_0485.MP4/00165.jpg"
#params = (0.53,0.34, -0.02, 7.8, 0.04)

im = load_im(file)
im = cv2.GaussianBlur(im, (7,7), 3,3)
H, topdown_im = topdown(im, params, should_warp=False)
Image.fromarray(topdown_im).save("image_with_box.png")

H, topdown_im = topdown(im, params, should_warp=True)
Image.fromarray(topdown_im.astype(np.uint8)).save("top_down_image.png")
# edges = ski.feature.canny(topdown_im, 2, 1, 25)
# lines = ski.transform.probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)
# plt.imshow(edges*0)
# for line in lines:
#     p0, p1 = line
#     plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
# # plt.imshow(edges)
# # hspace, angles, dist = ski.transform.hough_line(edges)
# plt.show()

#driver 182 hough line params
dst = cv2.Canny(np.uint8(topdown_im), 40, 50, None, 3)
canny = cv2.Canny(np.uint8(topdown_im), 40, 50, None, 3)
lines = cv2.HoughLines(dst, 1, np.pi / 180, 50, None, 0, 0)


#driver 23 hough params
# dst = cv2.Canny(np.uint8(topdown_im), 50, 200, None, 3)
# #cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# canny = cv2.Canny(np.uint8(topdown_im), 50, 200, None, 3)
# lines = cv2.HoughLines(dst, 1, np.pi / 180, 80, None, 0, 0)

cdst = cv2.cvtColor(np.uint8(im), cv2.COLOR_RGB2BGR)
cannydst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)


#print(lines)
Image.fromarray(canny[25:-25,25:-25]).show()
h_angles, h_lines = hough_lines(canny[:,50:-50], 80)
print(h_lines, h_angles)
if h_lines is not None:
    for i in range(0, len(h_lines)):
        rho = h_lines[i]+50
        theta = h_angles[i]
        if np.isclose(theta, 0, atol=0.1):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = crd_homog_to_cart(np.linalg.inv(H) @ np.array([int(x0 + 500*(-b)), int(y0 + 500*(a)) ,1]))
            pt2 = crd_homog_to_cart(np.linalg.inv(H) @ np.array([int(x0 - 1000*(-b)), int(y0 - 1000*(a)), 1]))
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
#            print(pt1, pt2)
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            # pt1 = crd_homog_to_cart(np.linalg.inv(H) @ np.array([int(x0 + 1000*(-b)), int(y0 + 1000*(a)) ,1]))
            # pt1 = (int(pt1[0]), int(pt1[1]))
            # cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cannydst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         if np.isclose(theta, 0, atol=0.1):
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             pt1 = crd_homog_to_cart(np.linalg.inv(H) @ np.array([int(x0 + 500*(-b)), int(y0 + 500*(a)) ,1]))
#             pt2 = crd_homog_to_cart(np.linalg.inv(H) @ np.array([int(x0 - 1000*(-b)), int(y0 - 1000*(a)), 1]))
#             pt1 = (int(pt1[0]), int(pt1[1]))
#             pt2 = (int(pt2[0]), int(pt2[1]))
# #            print(pt1, pt2)
#             cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
#             # pt1 = crd_homog_to_cart(np.linalg.inv(H) @ np.array([int(x0 + 1000*(-b)), int(y0 + 1000*(a)) ,1]))
#             # pt1 = (int(pt1[0]), int(pt1[1]))
#             # cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
#             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#             cv2.line(cannydst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

normal = cv2.cvtColor(cdst, cv2.COLOR_BGR2RGB)
Image.fromarray(normal).save("lines_warped_back.png")

normal = cv2.cvtColor(cannydst, cv2.COLOR_BGR2RGB)
Image.fromarray(normal).save("lines_on_canny.png")

normal =  cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
Image.fromarray(normal).save("canny.png")