import cv2
from pyzbar import pyzbar
import numpy as np

# 3D model points of the barcode corners in the real world.
# 3rd array is K so dont change it unless your qr is not planar
barcode_3d_points = np.array(
    [
        [0.0, 0.0],  # Bottom-left
        [6.0, 0.0],  # Bottom-right
        [6.0, 6.0],  # Top-right
        [0.0, 6.0],  # Top-left
    ],
    dtype=np.float32,
)


def calculate_distance(tvec):
    return np.linalg.norm(tvec)


def main():

    cap = cv2.VideoCapture(0)

    REAL_WIDTH = 5.0
    FOCAL_LENGTH_X = 717.4300715628018
    FOCAL_LENGTH_Y = 728.6036025661705

    # Camera matrix from calibration
    camera_matrix = np.array(
        [
            [FOCAL_LENGTH_X, 0, 524.81482166],
            [0, FOCAL_LENGTH_Y, 285.01862838],
            [0, 0, 1],
        ]
    )

    dist_coeffs = np.zeros((4, 1))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        barcodes = pyzbar.decode(frame)

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            barcode_2d_points = np.array(
                [
                    [x, y],  # Bottom-left
                    [x + w, y],  # Bottom-right
                    [x + w, y + h],  # Top-right
                    [x, y + h],  # Top-left
                ],
                dtype=np.float32,
            )

            # Compute homography
            H, _ = cv2.findHomography(barcode_3d_points, barcode_2d_points)

            # Decompose the homography matrix to get the rotation and translation vectors
            _, rvecs, tvecs, _ = cv2.decomposeHomographyMat(H, camera_matrix)

            # Use the first solution
            rvec = rvecs[0]
            tvec = tvecs[0]

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Calculate the distance to the barcode using the translation vector
            distance = calculate_distance(tvec)

            # Extract the yaw angle from the rotation matrix
            yaw_angle = (
                np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
            )

            cv2.putText(
                frame,
                f"Distance: {distance:.2f} cm",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Yaw Angle: {yaw_angle:.2f} degrees",
                (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
