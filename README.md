Camera QR Code Position Determination

This project aims to determine the position of a camera relative to a QR code or vice versa.
Dependencies

    OpenCV
    NumPy
    glob
    pyzbar
    math

Install the dependencies using pip:

    pip install opencv-python numpy glob3 pyzbar

Calibration

    Download all folders from the repository.
    Navigate to calibration.py and run it.
    The script will output the focal length (x, y) and camera matrix in the terminal. Note down these values for later use.

Running the System

    Open main.py. (Homography one does not work as intended)
    Enter the focal length (x, y) and camera matrix values obtained from the calibration step.
    Adjust the barcode_3d_points variable based on the dimensions of your QR code.
    If your QR code has different dimensions from 5cm, split the dimensions by 2 and enter them in the respective placeholders of the array.
    Run the system.

Some Images from project
![resim](https://github.com/AlpMercan/Positon-from-Qr/assets/112685013/9b799265-e4d4-4c1b-a500-9a310e697228)
