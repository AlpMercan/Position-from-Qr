import cv2
import numpy as np
import glob

# dimensions of the chessboard, do not touch unless you are trying something else
chessboard_size = (11, 7)

obj_points = []  # 3d points in real-world space
img_points = []  # 2d points in image plane

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)

images = glob.glob("chessboard_images/*.png")

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        obj_points.append(objp)
        img_points.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("Image", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

# Print the camera matrix
print("Camera matrix:")
print(camera_matrix)

# Extract the focal length
focal_length_x = camera_matrix[0, 0]
focal_length_y = camera_matrix[1, 1]

print("Focal length (x):", focal_length_x)
print("Focal length (y):", focal_length_y)
