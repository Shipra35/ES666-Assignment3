import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def detect_and_match_features(self, img1, img2):
        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Use BFMatcher with L2 norm for SIFT descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance (quality of match)
        matches = sorted(matches, key=lambda x: x.distance)

        return kp1, kp2, matches

    def estimate_homography(self, kp1, kp2, matches):
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Use Direct Linear Transform (DLT) to estimate homography
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0]
            u, v = dst_pts[i][0]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)

        return H / H[-1, -1]  # Normalize

    def stitch_images(self, img1, img2, H):
        # Warp the second image to align with the first image
        height, width = img1.shape[:2]
        warped_img2 = cv2.warpPerspective(img2, H, (width * 2, height))

        # Combine the two images into a panorama
        panorama = np.copy(warped_img2)
        panorama[0:height, 0:width] = img1

        return panorama

    def make_panaroma_for_images_in(self, path):
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} images for stitching.')

        stitched_image = cv2.imread(all_images[0])
        homography_matrix_list = []

        for i in range(1, len(all_images)):
            img1 = stitched_image
            img2 = cv2.imread(all_images[i])

            kp1, kp2, matches = self.detect_and_match_features(img1, img2)
            H = self.estimate_homography(kp1, kp2, matches)

            stitched_image = self.stitch_images(img1, img2, H)
            homography_matrix_list.append(H)

        return stitched_image, homography_matrix_list