import cv2
from pyzbar import pyzbar
import numpy as np
import math

# 3D model points of the barcode corners in the real world, centered at (0, 0, 0) for 5cm of qr code. just split the qr dimension by 2
# 3rd array is K so dont change it unless your qr is not planar
barcode_3d_points = np.array(
    [
        [-2.5, -2.5, 0.0],  # Bottom-left
        [2.5, -2.5, 0.0],  # Bottom-right
        [2.5, 2.5, 0.0],  # Top-right
        [-2.5, 2.5, 0.0],  # Top-left
    ],
    dtype=np.float32,
)


def calculate_theta_from_image(center, image_center, focal_length_x):
    dx = center[0] - image_center[0]
    dz = focal_length_x
    theta = np.arctan2(dx, dz) * 180 / np.pi
    return theta


def main():

    cap = cv2.VideoCapture(0)

    REAL_WIDTH = 5.0  # Placeholder for qr size cm
    FOCAL_LENGTH_X = 717.4300715628018  # derive this from calibration.py file
    FOCAL_LENGTH_Y = 728.6036025661705  # derive this from calibration.py file

    # Camera matrix from calibration, derive this from calibration.py file
    camera_matrix = np.array(
        [
            [FOCAL_LENGTH_X, 0, 524.81482166],
            [0, FOCAL_LENGTH_Y, 285.01862838],
            [0, 0, 1],
        ]
    )
    dist_coeffs = np.zeros((4, 1))
    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        barcodes = pyzbar.decode(frame)

        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Center of QR
            center = (x + w // 2, y + h // 2)
            image_center = (frame.shape[1] // 2, frame.shape[0] // 2)

            # SolvePnP to find rotation and translation vectors
            barcode_2d_points = np.array(
                [
                    [x, y],  # Bottom-left
                    [x + w, y],  # Bottom-right
                    [x + w, y + h],  # Top-right
                    [x, y + h],  # Top-left
                ],
                dtype=np.float32,
            )
            success, rvec, tvec = cv2.solvePnP(
                barcode_3d_points, barcode_2d_points, camera_matrix, dist_coeffs
            )
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(
                    rvec
                )  # ı thought ı may used that but nothing comes out

                # Extract x, y, z distances from tvec
                tvec = tvec.flatten()
                x_distance = tvec[0]
                y_distance = tvec[1]
                z_distance = tvec[2]

                # Calculate the angle theta from the image coordinates
                theta = calculate_theta_from_image(center, image_center, FOCAL_LENGTH_X)
                # Adjust x_distance based on theta
                Eculidian_dist = z_distance / math.cos(math.radians(theta))
                x_distance = z_distance * math.tan(math.radians(theta))

                # print(f"Center: {center}, Image Center: {image_center}")
                # print(f"Translation Vector (tvec): {tvec}")
                # print(
                #    f"X Distance: {x_distance}, Y Distance: {y_distance}, Z Distance: {z_distance}"
                # )
                # print(f"Theta: {theta}")

                # Marking base of image and qr, and line between them
                cv2.line(frame, image_center, center, (255, 0, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # Display x, y, z, theta and ecludian
                cv2.putText(
                    frame,
                    f"X: {x_distance:.2f} cm",
                    (x, y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Y: {y_distance:.2f} cm",
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Z: {z_distance:.2f} cm",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Theta: {theta:.2f} degrees",
                    (x, y - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"ALP: {Eculidian_dist:.2f} cm",
                    (x, y - 90),
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
