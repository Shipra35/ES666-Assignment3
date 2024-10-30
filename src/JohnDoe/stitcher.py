import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def detect_and_match_features(self, img1, img2):
        """Detect keypoints and match features using SIFT and BFMatcher."""
        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Match descriptors using BFMatcher (with L2 norm for SIFT)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        return kp1, kp2, matches

    def normalize_homography(self, H):
        """Normalize homography matrix so H[2,2] = 1."""
        return H / H[-1, -1]

    def estimate_homography(self, kp1, kp2, matches):
        """Estimate homography matrix using Direct Linear Transform (DLT)."""
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Construct matrix A for DLT
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0]
            u, v = dst_pts[i][0]
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])

        A = np.array(A)

        # Solve A * h = 0 using SVD (Singular Value Decomposition)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)  # The last row of V gives the solution

        return self.normalize_homography(H)

    def warp_image(self, img, H, output_shape):
        """Warp an image using a given homography matrix."""
        height, width = output_shape
        # Manually apply the homography matrix to warp the image
        warped_img = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                # Compute the original coordinates (inverse mapping)
                pt = np.array([j, i, 1])
                original_pt = np.dot(np.linalg.inv(H), pt)
                original_pt /= original_pt[-1]  # Normalize

                x, y = int(original_pt[0]), int(original_pt[1])

                # Copy pixel if within bounds
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    warped_img[i, j] = img[y, x]

        return warped_img

    def stitch_images(self, img1, img2, H):
        """Stitch two images using a homography matrix."""
        height, width = img1.shape[:2]

        # Warp the second image using the computed homography
        warped_img2 = self.warp_image(img2, H, (height, width * 2))

        # Copy the first image onto the warped canvas
        panorama = np.copy(warped_img2)
        panorama[0:height, 0:width] = img1

        return panorama

    def make_panaroma_for_images_in(self, path):
        """Create a panorama from all images in the given path."""
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} images for stitching.')

        # Start with the first image as the base
        stitched_image = cv2.imread(all_images[0])
        homography_matrix_list = []

        # Iterate through remaining images and stitch them
        for i in range(1, len(all_images)):
            img1 = stitched_image
            img2 = cv2.imread(all_images[i])

            # Detect and match features
            kp1, kp2, matches = self.detect_and_match_features(img1, img2)

            # Estimate homography matrix
            H = self.estimate_homography(kp1, kp2, matches)

            # Stitch images
            stitched_image = self.stitch_images(img1, img2, H)

            # Save homography matrix
            homography_matrix_list.append(H)

        return stitched_image, homography_matrix_list