# ANO

- [1. Segmentace obrazu](#1-segmentace-obrazu)
  - [1.1. Detekce hran](#11-detekce-hran)
  - [1.2. Metody průchodu nulou](#12-metody-průchodu-nulou)
  - [1.3. Cannyho detektor hran](#13-cannyho-detektor-hran)
- [2. OpenCV](#2-opencv)

Univerzální obrazové deskriptory - HoG

## 1. Segmentace obrazu

### 1.1. Detekce hran

Detekce oblastí stanovením hranice:

<img src="figures/edge-detection.png" alt="edge-detection" width="100px">

### 1.2. Metody průchodu nulou

Průběh jasu a jeho první a druhé derivace v místě hrany.

<img src="figures/brightness-and-derivations.png" alt="brightness-and-derivations" width="150px">

Velikost vektoru gradientů:

$$ \left\lVert \dfrac{\partial f(x,y)}{\partial x}, \dfrac{\partial f(x,y)}{\partial y} \right\rVert_2 $$

Laplacián (pro druhou derivaci):

$$ \dfrac{\partial^2 f(x,y)}{\partial x^2}, \dfrac{\partial^2 f(x,y)}{\partial y^2} $$

### 1.3. Cannyho detektor hran

Výsledná funkce vznikla minimalizací funkcionálu, který měří lokalizační chybu, signal2noise ratio (?)

## 2. OpenCV

Load image:

```cpp
cv::Mat src_8uc3_img = cv::imread( "images/lena.png", cv::IMREAD_COLOR );
cv::Mat src_8uc1_img = cv::imread( "images/lena.png", cv::IMREAD_GRAYSCALE );
```

- One pixel is represented by unsigned char (8 bits).

Conversion to gryscale:

```cpp
cv::cvtColor( src_8uc3_img, gray_8uc1_img, cv::COLOR_BGR2GRAY );
```

Conversion to float $([0,255] \rightarrow [0,1])$:

```cpp
gray_8uc1_img.convertTo( gray_32fc1_img, CV_32FC1, 1.0 / 255.0 );
```

Draw a rectangle:

```cpp
cv::rectangle(gray_8uc1_img,
              cv::Point(65, 84),
              cv::Point(75, 94),
              cv::Scalar(50),
              cv::FILLED);
```

<details><summary> Example: Access pixel values </summary>

- template method `cv::Mat.at<image_type>(int y, int x)`

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    int x = 0;
    int y = 0;

    // read grayscale value of a pixel, image represented using 8 bits
    uchar p1 = gray_8uc1_img.at<uchar>(y, x);

    // read grayscale value of a pixel, image represented using 32 bits
    float p2 = gray_32fc1_img.at<float>(y, x);

    // read color value of a pixel, image represented using 8 bits per color channel
    cv::Vec3b p3 = src_8uc3_img.at<cv::Vec3b>(y, x);

    // print values of pixels
    printf("p1 = %d\n", p1);
    printf("p2 = %f\n", p2);
    printf("p3[0] = %d, p3[1] = %d, p3[2] = %d\n", p3[0], p3[1], p3[2]);

    // set pixel value to 0 (black)
    gray_8uc1_img.at<uchar>( y, x ) = 0;

    return 0;
}
```

</details>

<details><summary> Example: Creating gradient </summary>

```cpp
// Declare a variable to hold the gradient image with dimensions:
// width = 256 pixels, height = 50 pixels.
// Gray levels wil be represented using 8 bits (uchar).
cv::Mat gradient_8uc1_img( 50, 256, CV_8UC1 );

// For every pixel in image, 
// assign a brightness value
// according to the `x` coordinate.
// This wil create a horizontal gradient.
for ( int y = 0; y < gradient_8uc1_img.rows; y++ ) {
    for ( int x = 0; x < gradient_8uc1_img.cols; x++ ) {
        gradient_8uc1_img.at<uchar>( y, x ) = x;
    }
}

cv::imshow("Gradient", gradient_8uc1_img);
```

</details>
