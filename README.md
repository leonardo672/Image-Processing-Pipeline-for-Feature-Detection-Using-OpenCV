# Image-Processing-Pipeline-for-Feature-Detection-Using-OpenCV

#### This project implements an advanced image processing pipeline using OpenCV. The pipeline performs a series of transformations and feature detection tasks on an input image to highlight and analyze key features.


## Key Features:
### Original Image Display: Provides a baseline view of the input image.
![Image](https://github.com/user-attachments/assets/0a0f092d-1124-4d59-9554-ffbcadd72163)

### Grayscale Conversion: Simplifies the image by removing color information, making it easier to process.
![Image_Gray](https://github.com/user-attachments/assets/9d20cf14-9f29-4f43-8321-73a1965f8718)

### Gaussian Blurring: Reduces noise and detail, smoothing the image for better edge detection.
![Image_Blur](https://github.com/user-attachments/assets/1f05a19f-41f7-4359-8732-6cba7cf077c3)

### Edge Detection: Utilizes the Canny algorithm to identify and highlight edges within the image.
![Image_Canny](https://github.com/user-attachments/assets/c17de9ed-4d4b-4e83-890e-053fa6932f85)

## Morphological Operations:
### Dilation: Expands detected edges to enhance their visibility.
![Image_Dilation](https://github.com/user-attachments/assets/f7f126f6-6c23-4b9f-8d94-7c7f9939be9e)

### Erosion: Refines the edges by removing small noise and smoothing.
![Image_Erode](https://github.com/user-attachments/assets/29dbe26f-4dda-4498-ad22-7256b2a8d51a)

### Contour Detection: Identifies and highlights shapes or objects within the image by drawing contours around them.
![Contours](https://github.com/user-attachments/assets/7f5d914c-6f10-49ab-a890-ebee327c109b)

### Adaptive Thresholding: Separates objects from the background under varying lighting conditions.
![Adaptive_Threshold](https://github.com/user-attachments/assets/501e8b5e-a62b-4177-a07b-3a4ddf65ed9b)

### Face Detection: Uses a pre-trained Haar Cascade classifier to detect and mark faces in the image.
![Detected_Faces](https://github.com/user-attachments/assets/a0e02308-516b-4aff-afe3-34185b338a75)


