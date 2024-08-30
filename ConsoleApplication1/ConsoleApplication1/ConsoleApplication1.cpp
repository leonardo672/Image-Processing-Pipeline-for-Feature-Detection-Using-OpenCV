#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// Define the path for saving processed images
const string saveFolder = "FACE_DETECTED";

// Global variables for trackbars
int lowThreshold = 50, highThreshold = 100;
int blurKernelSize = 3;  // Specifically for GaussianBlur
Mat img, imgGray, imgBlur, imgCanny, imgDilation, imgErode, imgContours, imgAdaptiveThresh, imgDetectedFaces;
Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

// Function to save images to the designated folder
void saveImages() {
    // Save each image with appropriate filenames
    if (!img.empty()) imwrite(saveFolder + "/Image.jpg", img);
    if (!imgGray.empty()) imwrite(saveFolder + "/Image_Gray.jpg", imgGray);
    if (!imgBlur.empty()) imwrite(saveFolder + "/Image_Blur.jpg", imgBlur);
    if (!imgCanny.empty()) imwrite(saveFolder + "/Image_Canny.jpg", imgCanny);
    if (!imgDilation.empty()) imwrite(saveFolder + "/Image_Dilation.jpg", imgDilation);
    if (!imgErode.empty()) imwrite(saveFolder + "/Image_Erode.jpg", imgErode);
    if (!imgContours.empty()) imwrite(saveFolder + "/Contours.jpg", imgContours);
    if (!imgDetectedFaces.empty()) imwrite(saveFolder + "/Detected_Faces.jpg", imgDetectedFaces);
    if (!imgAdaptiveThresh.empty()) imwrite(saveFolder + "/Adaptive_Threshold.jpg", imgAdaptiveThresh);

    cout << "All images have been saved to the 'FACE_DETECTED' folder." << endl;
}

// Trackbar callback function
void onTrackbarChange(int, void*) {
    // Apply GaussianBlur with the adjustable kernel size
    int actualKernelSize = blurKernelSize * 2 + 1;  // Ensure kernel size is odd
    GaussianBlur(imgGray, imgBlur, Size(actualKernelSize, actualKernelSize), 0, 0);

    // Canny Edge Detection with adjustable thresholds
    Canny(imgBlur, imgCanny, lowThreshold, highThreshold);

    // Dilation and Erosion
    dilate(imgCanny, imgDilation, kernel);
    erode(imgDilation, imgErode, kernel);

    // Contour Detection
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgCanny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    imgContours = img.clone();
    drawContours(imgContours, contours, -1, Scalar(0, 255, 0), 2);

    // Hough Transform for Line Detection
    vector<Vec4i> lines;
    HoughLinesP(imgCanny, lines, 1, CV_PI / 180, 50, 50, 10);
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(imgContours, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
    }

    // Display images
    imshow("Image Canny", imgCanny);
    imshow("Image Dilation", imgDilation);
    imshow("Image Erode", imgErode);
    imshow("Contours", imgContours);

    // Save the updated images
    saveImages();
}

// Function to display help information
void displayHelp() {
    cout << "------------------- HELP -------------------" << endl;
    cout << "This application provides the following functionalities:" << endl;
    cout << "1. Edge Detection (Canny): Adjust thresholds using trackbars." << endl;
    cout << "2. Contour Detection: Displays contours of objects." << endl;
    cout << "3. Hough Line Transform: Detects and draws lines in the image." << endl;
    cout << "4. Face Detection: Detects faces using a pre-trained Haar Cascade." << endl;
    cout << "5. Adaptive Thresholding: Handles varying lighting conditions." << endl;
    cout << "6. Automatic Saving: All processed images are automatically saved to 'FACE_DETECTED' folder." << endl;
    cout << "7. Manual Save: Press 's' to save all processed images manually." << endl;
    cout << "8. Exit: Press 'ESC' key to exit the application." << endl;
    cout << "---------------------------------------------" << endl;
}

int main() {
    // Display help information
    displayHelp();

    // Load the image
    string path = "istockphoto-507995592-612x612.jpg";
    img = imread(path);
    if (img.empty()) {
        cout << "Could not open or find the image at path: " << path << endl;
        return -1;
    }

    // Convert to grayscale and blur the image
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgBlur, Size(7, 7), 0, 0);

    // Adaptive Thresholding
    adaptiveThreshold(imgGray, imgAdaptiveThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    imshow("Adaptive Threshold", imgAdaptiveThresh);

    // Save Adaptive Threshold image
    saveImages();  // This will save all images, including Adaptive Threshold

    // Face Detection using Haar Cascade
    CascadeClassifier faceCascade;
    // Ensure the Haar Cascade file is in the correct path
    string faceCascadePath = "haarcascade_frontalface_default.xml";
    if (!faceCascade.load(faceCascadePath)) {
        cout << "Error loading Haar Cascade file from path: " << faceCascadePath << endl;
    }
    else {
        vector<Rect> faces;
        faceCascade.detectMultiScale(imgGray, faces, 1.1, 10);
        imgDetectedFaces = img.clone();
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(imgDetectedFaces, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
        }
        imshow("Detected Faces", imgDetectedFaces);
    }

    // Save Detected Faces image
    saveImages();  // This will save all images, including Detected Faces

    // Display original and gray images
    imshow("Image", img);
    imshow("Image Gray", imgGray);
    imshow("Image Blur", imgBlur);

    // Save original, gray, and blur images
    saveImages();  // This will save all images, including original, gray, and blur

    // Trackbars for Canny edge detection parameters
    namedWindow("Image Canny", WINDOW_AUTOSIZE);
    namedWindow("Image Dilation", WINDOW_AUTOSIZE);
    namedWindow("Image Erode", WINDOW_AUTOSIZE);
    namedWindow("Contours", WINDOW_AUTOSIZE);

    createTrackbar("Low Threshold", "Image Canny", &lowThreshold, 100, onTrackbarChange);
    createTrackbar("High Threshold", "Image Canny", &highThreshold, 200, onTrackbarChange);
    createTrackbar("Blur Kernel Size", "Image Blur", &blurKernelSize, 7, onTrackbarChange);

    // Initial call to the trackbar callback function to display and save images
    onTrackbarChange(0, 0);

    // Main loop to handle key events
    while (true) {
        int key = waitKey(1);

        if (key == 27) {  // ESC key to exit
            cout << "Exiting application." << endl;
            break;
        }
        else if (key == 's' || key == 'S') {  // 's' key to save images manually
            saveImages();
        }
    }

    return 0;
}
