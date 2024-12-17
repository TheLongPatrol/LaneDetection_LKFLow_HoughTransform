from typing import List
import math
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import statistics

from scipy.cluster.vq import vq, kmeans, whiten, kmeans2

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
    transform_HS = transform_H + transform_S
    transform_inv = transform_HS.inverse

    result_shape = off_max - off_min

    im_warp = ski.transform.warp(im, transform_inv, preserve_range=True, output_shape=result_shape)

    return im_warp, transform_HS

def topdown(im, params, box_dim=400):
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

    warp_im, transform = warp(im, box_coords, H)
    warp_im = warp_im.astype(np.uint8)

    im[ski.draw.line(*box_coords[0], *box_coords[1])[::-1]] = 255
    im[ski.draw.line(*box_coords[1], *box_coords[2])[::-1]] = 255
    im[ski.draw.line(*box_coords[2], *box_coords[3])[::-1]] = 255
    im[ski.draw.line(*box_coords[3], *box_coords[0])[::-1]] = 255

    return H, im, warp_im, transform

def crd_cart_to_polar(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    phi = np.arctan2(y2 - y1, x2 - x1)
    theta = phi + np.pi / 2

    rho = np.abs(x2 * y1 - x1 * y2) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return rho, theta

def reference_lines(ref_fname, transform):
    def line_to_polar(line):
        coords = [float(v) for v in line.strip().split(' ')]

        p1 = transform(coords[:2])[0]
        p2 = transform(coords[-2:])[0]

        return crd_cart_to_polar(p1, p2)

    with open(ref_fname, 'r') as ref_file:
        return np.array([line_to_polar(line) for line in ref_file.readlines()])

def hough_lines(img, threshold):
    r_max = math.ceil(np.sqrt(img.shape[0]**2+img.shape[1]**2))
    accumulator = np.zeros(shape=(360, r_max))
    angles = np.arange(360)*np.pi/180
    sines = np.sin(angles)
    cosines = np.cos(angles)
    angle_inds = np.arange(360)
    angle_r_to_x_y = [[[] for _ in range(r_max)] for _ in range(360)]
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y,x] == 255:
                rs = np.round(x*cosines + y*sines).astype(int)
                # print(rs.shape, sines.shape, rs.dtype)
                # print(rs)
                accumulator[angle_inds, rs]+=1
                for i in range(360):
                    angle_r_to_x_y[i][rs[i]].append([x,y])
   # print(accumulator.mean())
    #print(accumulator)
    #thetas, rs, = np.where(accumulator>threshold)
    window_rad = 10
    filtered_acc = np.zeros(shape=accumulator.shape)
    for r in range(r_max):
        left_bound = max(0, r - window_rad)
        right_bound = min(r_max-1, r+window_rad)
        for i in range(5):
            ang_lower = max(0, i-2)
            ang_upper = i+2
            if accumulator[i][r] == np.max(accumulator[ang_lower:ang_upper, left_bound:right_bound]):
                filtered_acc[i][r] = accumulator[i][r]
                accumulator[ang_lower:ang_upper, left_bound:right_bound] = 0
        for i in range(178, 183):
            ang_lower = max(0, i-2)
            ang_upper = i+2
            if accumulator[i][r] == np.max(accumulator[ang_lower:ang_upper, left_bound:right_bound]):
                filtered_acc[i][r] = accumulator[i][r]
                accumulator[ang_lower:ang_upper, left_bound:right_bound] = 0
        for i in range(354,360):
            ang_lower = i-2
            ang_upper = min(359, i+2)
            if accumulator[i][r] == np.max(accumulator[ang_lower:ang_upper, left_bound:right_bound]):
                filtered_acc[i][r] = accumulator[i][r]
                accumulator[ang_lower:ang_upper, left_bound:right_bound] = 0
    thetas, rs = np.where(filtered_acc>threshold)
    # print(np.array(angle_r_to_x_y, dtype=object).shape, np.where(rs == 532))
    # print(np.array(angle_r_to_x_y, dtype=object)[:, 532])
    # return
    filtered_angle_r_to_x_y = np.empty(shape=filtered_acc.shape, dtype=object)
    #print(np.array(angle_r_to_x_y, dtype=object)[thetas, rs])
    filtered_angle_r_to_x_y[thetas,rs] = np.array(angle_r_to_x_y, dtype=object)[thetas, rs]
    # print(thetas, rs)
    # print(filtered_acc[thetas, rs])
    
    # thetas, rs = np.where(filtered_acc >=  threshold)
    
#    accumulator = accumulator[:,3:]
    #thetas, rs = np.nonzero(np.where(accumulator >  0.5*np.max(accumulator)))
    return  np.vstack((np.arange(r_max)[rs], thetas)).T, filtered_angle_r_to_x_y

def optical_flow(topdown_im_0,topdown_im_1,p0):

    # print(p0)   
    p0 = p0.reshape((p0.shape[0], 1,p0.shape[1])) 
    

    lk_params = dict( winSize  = (20, 20),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 1,
                       blockSize = 1 )
    # print(p0.shape)
    # p0 = cv2.goodFeaturesToTrack(topdown_im_0.astype('uint8'), mask = None, **feature_params)
    # print(p0)
    # p0 = p0[p0[:,0,1] < 350]

    p1, st, err = cv2.calcOpticalFlowPyrLK(topdown_im_0.astype('uint8'), topdown_im_1.astype('uint8'), p0.astype('float32'), None, **lk_params)
    p0 = p0.reshape(p0.shape[0], -1)
    p1 = p1.reshape(p1.shape[0], -1)

    # change in location of points
    mvmt = p1-p0
    mag, ang = cv2.cartToPolar(mvmt[..., 0], mvmt[..., 1])
    # print(ang.shape)

    std_dev_ang = statistics.stdev(ang[:,0])
    std_dev_mag = statistics.stdev(mag[:,0])

    #filter based on standard deviation
    filter = ((ang[:,0] >= np.mean(ang[:,0]) - std_dev_ang) & 
                     (ang[:,0] <= np.mean(ang[:,0]) + std_dev_ang))
    
    filter_mag = ((mag[:,0] >= np.mean(mag[:,0]) - std_dev_mag) & 
                    (mag[:,0] <= np.mean(mag[:,0]) + std_dev_mag))
    
    p0 = p0[filter & filter_mag,:]
    p1 = p1[filter & filter_mag,:]
    mvmt = mvmt[filter & filter_mag,:]
    mag, ang = cv2.cartToPolar(mvmt[..., 0], mvmt[..., 1])
    shift = np.mean((p1-p0)[:,0])
    # print(np.mean(ang))

    # p0 = p0[filter_mag,:]
    # p1 = p1[filter_mag,:]
    # mvmt = mvmt[filter_mag,:]
    # p0 = p0[(ang >= np.mean(ang) - std_dev_ang) & 
    #                  (ang <= np.mean(ang) + std_dev_ang),:]
    


    # print(mag,ang)
    mvmt_mag = np.linalg.norm(p1-p0, axis = 1)

    fig, ax = plt.subplots()
    ax.quiver(p0[:,0], p0[:,1], mvmt[:,0], mvmt[:,1], angles='xy', scale_units='xy', scale=1, color='r')

    # plt.scatter(p1[:,0],p1[:,1])
    # img = mpimg.imread('/Users/dbelgorod/Documents/UIUC/Fall_2024/CS543/Project/driver_100_30frame/05250653_0338.MP4/00390.jpg')
    ax.imshow(topdown_im_0,cmap='gray')
    plt.show()
    # print(shift)

    return shift

def get_car_direction(im0, im1):

    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.6,
                        minDistance = 7,
                        blockSize = 9 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(im0.astype('uint8'), mask = None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(im0.astype('uint8'), im1.astype('uint8'), p0, None, **lk_params)
    p0 = p0.reshape(p0.shape[0], -1)
    p1 = p1.reshape(p1.shape[0], -1)
    # color = (255,0, 0)  # Green color in BGR format

    # remove all points below the dashboard
    dash = 400

    # p1 = p1[p1[:,1] < 350]

    # change in location of points
    mvmt = p1-p0
    mag, ang = cv2.cartToPolar(mvmt[..., 0], mvmt[..., 1])
    # print(mag,ang)
    mvmt_mag = np.linalg.norm(p1-p0, axis = 1)

    #calculate relative motion from center distance
    origin = np.array([int(im0.shape[1]/2),775])

    # determine the type of motion
    # determine if all have a component in the horiz moving in one direction
    mvmt_pos = sum(mvmt[:,0]> 0) 
    mvmt_neg = sum(mvmt[:,0]< 0) 

    fwd = True
    if mvmt.shape[0]>0:
        if (mvmt_pos/mvmt.shape[0]>0.3) & (mvmt_pos/mvmt.shape[0]<0.7):
            fwd = True
        else:
            fwd = False

        # if move in one direction, check if changing lanes or turning
        # if turning, check what direction
        right = False

        if not fwd:
            if mvmt_pos > mvmt_neg:
                right = True
            else:
                right = False


        # logic for lane change ?

        # assume small motion
        pnt_dist = p0 - origin
        mag_0, ang_0 = cv2.cartToPolar(pnt_dist[..., 0], pnt_dist[..., 1])



        # partition into right/left sides
        right_pts_mask = pnt_dist[:,0] > 0 
        left_pts = pnt_dist[:,0] < 0 

        # filter out large angular distances irregularities
        # do it based on movement
        # if turning, filter out all not going in that direction
        # if not turning, filter out based on all not pointing away from center
        # print(180*(ang-ang_0)[right_pts_mask]/np.pi)


        # filter out large magnatude change
        # print(p0)
        # print(pnt_dist)


        fig, ax = plt.subplots()
        ax.quiver(p0[:,0], p0[:,1], mvmt[:,0], mvmt[:,1], angles='xy', scale_units='xy', scale=1, color='r')

        # plt.scatter(p1[:,0],p1[:,1])
        # img = mpimg.imread('/Users/dbelgorod/Documents/UIUC/Fall_2024/CS543/Project/driver_100_30frame/05250653_0338.MP4/00390.jpg')
        ax.imshow(im0,cmap='gray')
        plt.show()

        # fig, ax = plt.subplots()
        # ax.imshow(im0,cmap='gray')

        # fig, ax = plt.subplots()
        # ax.imshow(im1,cmap='gray')
    pass

def plot_lines_from_points(cdst, cannydst, prev_lines, transform, output_prefix, output_postfix):
    for line in prev_lines:
        for i in range(len(line)-1):
            cv2.line(cannydst, line[i], line[i+1], (0,0,255), 3, cv2.LINE_AA)
            pt1 = transform.inverse(crd_homog_to_cart(np.array([line[i][0], line[i][1],1])))[0]
            pt2 = transform.inverse(crd_homog_to_cart(np.array([line[i+1][0], line[i+1][1], 1])))[0]
            print(pt1)
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    normal = cv2.cvtColor(cdst, cv2.COLOR_BGR2RGB)
    Image.fromarray(normal).save(output_prefix+"lines_warped_back"+output_postfix)

    normal = cv2.cvtColor(cannydst, cv2.COLOR_BGR2RGB)
    Image.fromarray(normal).save(output_prefix+"lines_on_canny"+output_postfix)



def get_optical_flow_lines(im, prev_im, prev_im_orig, unprocessed_topdown, prev_pts, prev_lines):
    dir = get_car_direction(im, prev_im_orig)
    shift = optical_flow(prev_im,unprocessed_topdown,prev_pts[1:,:])
    shift_lines = []
    for line in prev_lines:
        temp_line = np.array(line[:])
        temp_line[:,0] = temp_line[:,0] + shift
        shift_lines.append(temp_line.tolist())
    return shift_lines
def generate_outputs_for_driver(driver_path, output_dir, homog_params):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    direct = sorted(os.listdir(driver))
    for video in direct[30:31]:
        prev_lines = []
        if not os.path.isdir(output_dir+video):
            os.mkdir(output_dir+video)
        files = [frame for frame in os.listdir(driver_path+video) if len(frame.split("."))>1 and frame.split(".")[-1] !="txt"]
        files= sorted(files)
        prev_im_orig = np.zeros((1,1))
        prev_im = np.zeros((1,1)) # holder for previous image array
        prev_pts = np.zeros((1,2)) # holder for previous
        for file in files:
            filename = driver_path+video+"/"+file
            output_prefix = output_dir+video+"/"+file.split(".")[0]+"_"
            output_postfix = ".png"
            #Load file and do homography
            im = load_im(filename)
            H, box_im, topdown_im, transform = topdown(im, homog_params)
#            Image.fromarray(topdown_im).save(output_prefix+"topdown.png")
            unprocessed_topdown = np.copy(topdown_im)
            #Histogram
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            topdown_im = clahe.apply(topdown_im)

            topdown_im = cv2.GaussianBlur(topdown_im, (17, 17), 3, 3)
            #Setup canny images
            cdst = cv2.Canny(np.uint8(topdown_im), 40, 50, None, 3)
            canny = cv2.Canny(np.uint8(topdown_im), 40, 50, None, 3)
            hough = np.copy(canny)
            cdst = cv2.cvtColor(np.uint8(im), cv2.COLOR_RGB2BGR)
            cannydst = cv2.cvtColor(hough, cv2.COLOR_GRAY2BGR)
            #Compute Hough Transform
            lines, angle_r_to_x_y = hough_lines(canny, 30)
            near_zero = lines[lines[:, 1] < 2]
            two_pi_lines = lines[np.abs(lines[:,1] - 360) < 2]
            two_pi_copy = np.copy(two_pi_lines)
            two_pi_lines[:,1] = two_pi_lines[:,1] - 360
            
            lines = np.vstack((near_zero, two_pi_lines))
            lines_act = np.vstack((near_zero, two_pi_copy))
            #Check if at least 4 lines, so kmeans doesn't give error
            if len(lines)>3:
                centers, _ = kmeans(lines.astype(float), 4)
                #centers, _ = kmeans2(lines, 4, minit="++")
                #Check if 4 centers, if so Hough Lines
              #  if len(centers)>3:
                    #Get actual lines closest to cluster center
                cur_lines = []
                best_lines = np.zeros(shape=centers.shape)
                good_lines = True
                for i in range(len(centers)):
                    dists_to_cluster = np.linalg.norm(lines - centers[i], axis=1)
                    ind = np.argmin(dists_to_cluster)
                    best_lines[i] = lines[ind]
                    act_line = lines_act[ind]
                    # if dists_to_cluster.min() > 0:
                    #     ind = np.argmin(dists_to_cluster)
                    #     best_lines[i] = lines[ind]
                    #     act_line = lines_act[ind]
                    # else:
                    #     inds = np.where(dists_to_cluster < 10)
                    #     fil_lines = lines_act[inds]
                    #     temp = np.array(list(map(len, angle_r_to_x_y[fil_lines[:,1], fil_lines[:,0]])))
                    #     ind = np.argmax(temp)
                    #     best_lines[i] = fil_lines[ind]
                    #     act_line = fil_lines[ind]
               #     print(act_line)
                    pts = angle_r_to_x_y[act_line[1], act_line[0]]
                    if len(pts) < 35:
                        good_lines = False
                        break
                    cur_lines.append(pts)
                if good_lines:
                    #Plot lines
                    prev_lines = cur_lines
                    for line in best_lines:
                        rho = line[0]
                        theta = line[1]*np.pi/180

                        a = math.cos(theta)
                        b = math.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    
                        cv2.line(topdown_im, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)
                #        if np.isclose(theta, 0, atol=0.1):
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
                        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    #    cv2.line(cannydst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                    for line in prev_lines:
                        for i in range(len(line)-1):
                            cv2.line(cannydst, line[i], line[i+1], (0,0,255), 3, cv2.LINE_AA)
                    Image.fromarray(topdown_im).save(output_prefix+"lines_on_top_down"+output_postfix)
                    normal = cv2.cvtColor(cdst, cv2.COLOR_BGR2RGB)
                    Image.fromarray(normal).save(output_prefix+"lines_warped_back"+output_postfix)

                    normal = cv2.cvtColor(cannydst, cv2.COLOR_BGR2RGB)
                    Image.fromarray(normal).save(output_prefix+"lines_on_canny"+output_postfix)

                #
                else:
                    if len(prev_pts) > 1: # if statement so it doesnt apply to the first image  
                        prev_lines = get_optical_flow_lines(im, prev_im, prev_im_orig, unprocessed_topdown, prev_pts, prev_lines)
                        
                        best_lines = np.zeros(shape=(len(prev_lines), 2))
                        for i in range(len(prev_lines)):
                            best_lines[i] = crd_cart_to_polar(prev_lines[i][0], prev_lines[i][-1])
                        plot_lines_from_points(cdst, cannydst, prev_lines, transform, output_prefix, output_postfix)

            else:
                if len(prev_pts)>1:
                    prev_lines = get_optical_flow_lines(im, prev_im, prev_im_orig, unprocessed_topdown, prev_pts, prev_lines)
        #            print(prev_lines)
                    best_lines = np.zeros(shape=(len(prev_lines), 2))
                    for i in range(len(prev_lines)):
                        best_lines[i] = crd_cart_to_polar(prev_lines[i][0], prev_lines[i][-1])
                    plot_lines_from_points(cdst, cannydst, prev_lines, transform, output_prefix, output_postfix)
                        # convert prev_lines to points
            prev_pts = np.array([pt for line in prev_lines for pt in line])
            prev_im = unprocessed_topdown
            prev_im_orig = im
            
                
            # else:  


                    # normal =  cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
                    # Image.fromarray(normal).save(output_prefix+"canny.png")

# in source data, lane width -> 3.75m, marker length -> 3m



# driver161

#file = "CULane/driver_161_90frame/06030828_0758.MP4/00630.jpg"
# params = (0.7, 0.1, -0.035)

#driver 182
# file = "CULane/driver_182_30frame/06011750_0258.MP4/02040.jpg"
# file = "CULane/driver_182_30frame/06011844_0276.MP4/00030.jpg"
#file = "CULane/driver_182_30frame/05312327_0001.MP4/02160.jpg"

#file = "CULane/driver_182_30frame/06011702_0242.MP4/02040.jpg"
#file = "CULane/driver_182_30frame/06011844_0276.MP4/00030.jpg"
#file = "CULane/driver_182_30frame/06011750_0258.MP4/02040.jpg"
file = "CULane/driver_182_30frame/06010816_0097.MP4/00390.jpg"
params = (0.51, 0.32, -0.03, 7.4, 0.05)

#driver 23
#file = "CULane/driver_23_30frame/05170623_0673.MP4/00020.jpg"
#file = "CULane/driver_23_30frame/05170732_0696.MP4/00200.jpg"
#file = "CULane/driver_23_30frame/05160923_0485.MP4/00165.jpg"
#params = (0.53,0.34, -0.02, 7.8, 0.04)


driver = "./CULane/driver_182_30frame/"
output_dir = "./driver_182_30frame_outputs/"
generate_outputs_for_driver(driver, output_dir, params)
