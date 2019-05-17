//
//  main.cpp
//  cameraCalibration
//
//  Created by Junjie He on 25/04/19.
//  Copyright © 2019 Junjie He. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>


#define COLOR (0,0,255)
using namespace cv;
using namespace std;

    /// Global variables
vector<Point> pointBank;
vector<Point> sortBank,pureBank;
Mat src, src_gray,reSrc;
Mat myHarris_dst, myHarris_copy, Mc;
Mat myShiTomasi_dst, myShiTomasi_copy;
const String img_1_Address = "/Users/chegg/workShop/computerVision/CameraCalibration/cameraCalibration/cameraCalibration/img1.jpg";
const String img_2_Address = "/Users/chegg/workShop/computerVision/CameraCalibration/cameraCalibration/cameraCalibration/img2.jpg";
const String img_1_Save1Address = "/Users/chegg/workShop/computerVision/CameraCalibration/cameraCalibration/cameraCalibration/img1_coordinate.jpg";
const String img_2_Save2Address = "/Users/chegg/workShop/computerVision/CameraCalibration/cameraCalibration/cameraCalibration/img2_coordinate.jpg";

const String imgSrc= img_2_Address;
const String imgSave= img_2_Save2Address;

int myShiTomasi_qualityLevel = 50;
//int myHarris_qualityLevel = 50;
int max_qualityLevel = 100;

//double myHarris_minVal, myHarris_maxVal;
double myShiTomasi_minVal, myShiTomasi_maxVal;

RNG rng(12345);

const char* myHarris_window = "My Harris corner detector";
const char* myShiTomasi_window = "My Shi Tomasi corner detector";
//
//    /// Slide bar
//const int alpha_slider_max = 100;
//int alpha_slider;
//double alpha;
//double beta;

    /// Function headers
void myShiTomasi_function( int, void* );
void myHarris_function( int, void* );
void printPoint_function(void);

/**
 * @function main
 */
int main( int argc, char** argv )
{
   
        /// Load source image and convert it to gray
    src = imread(imgSrc);
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    
        /// Set some parameters
    int blockSize = 3, apertureSize = 3;
    
        /// My Harris matrix -- Using cornerEigenValsAndVecs
    cornerEigenValsAndVecs( src_gray, myHarris_dst, blockSize, apertureSize );
    
    /* calculate Mc */
    Mc = Mat( src_gray.size(), CV_32FC1 );
    for( int i = 0; i < src_gray.rows; i++ )
    {
        for( int j = 0; j < src_gray.cols; j++ )
        {
            float lambda_1 = myHarris_dst.at<Vec6f>(i, j)[0];
            float lambda_2 = myHarris_dst.at<Vec6f>(i, j)[1];
            Mc.at<float>(i, j) = lambda_1*lambda_2 - 0.04f*pow( ( lambda_1 + lambda_2 ), 2 );
        }
    }
    
//    minMaxLoc( Mc, &myHarris_minVal, &myHarris_maxVal );
    
        /// My Shi-Tomasi -- Using cornerMinEigenVal
    cornerMinEigenVal( src_gray, myShiTomasi_dst, blockSize, apertureSize );
    
    minMaxLoc( myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal );
    
    /* Create Window and Trackbar */
    namedWindow( myShiTomasi_window ,cv::WINDOW_AUTOSIZE);
    resizeWindow(myShiTomasi_window, 3000, 2250);
    createTrackbar( "Quality Level:", myShiTomasi_window, &myShiTomasi_qualityLevel, max_qualityLevel, myShiTomasi_function );
    myShiTomasi_function( 0, 0 );
    
    while (1)
    {
        char key = cvWaitKey();
        printPoint_function();
        if (key == '0')
            return 0 ;
    }
}
bool comparePoint(Point pt1, Point pt2)
{
    return (pt1.x < pt2.x);
}
bool comparePointY(Point pt1, Point pt2)
{
    return (pt1.y < pt2.y);
}

/* function name: printPoint_function
 * description:
 * @param <#param#>:
 */
void printPoint_function(void)
{
    //print the coordinate with circle
    Point lastPt;
    sort(pointBank.begin(),pointBank.end(),comparePoint);
    pointBank.erase( unique( pointBank.begin(),pointBank.end() ), pointBank.end() );
//    cout<<"orig:"<<pointBank<<endl;

    vector<int> eraseList;
    sortBank.push_back(pointBank[0]);
    sort(pointBank.begin(),pointBank.end(),comparePoint);
    for (int itr = 1; itr != pointBank.size(); ++itr)
    {
        if (pointBank.at(itr).x == lastPt.x)
        {
            if (abs(pointBank.at(itr).y - lastPt.y)>10)
            {
                sortBank.push_back(pointBank.at(itr));
            }
        }else{
//            if ( abs(pointBank.at(itr).x - lastPt.x) > 20 && abs(pointBank.at(itr).y - lastPt.y) >10)
                sortBank.push_back(pointBank.at(itr));
        }
        lastPt =pointBank.at(itr);
    }
    sort(sortBank.begin(),sortBank.end(),comparePoint);
    
    for( Point pt : sortBank)
    {
        cv::circle(myShiTomasi_copy, pt, 3, COLOR);
        std::string content= std::to_string(pt.x) + "," + std::to_string(pt.y);
        cv::putText(myShiTomasi_copy, content, pt,
                    cv::FONT_HERSHEY_SIMPLEX, .5, COLOR,.3,
                    cv::LineTypes::LINE_4);
    }
    //finally show changes on image
    imwrite(imgSave, myShiTomasi_copy);
    cv::resize(myShiTomasi_copy, myShiTomasi_copy, cv::Size(),0.5,0.5);
    resizeWindow(myShiTomasi_window, 3000/2, 2250/2);
    imshow( myShiTomasi_window, myShiTomasi_copy );
    pureBank.clear();
    sortBank.clear();
    pointBank.clear();
}
/**
 * @function myShiTomasi_function
 */
void myShiTomasi_function( int, void* )
{
    myShiTomasi_copy = src.clone();
    myShiTomasi_qualityLevel = MAX(myShiTomasi_qualityLevel, 1);
    pureBank.clear();
    sortBank.clear();
    pointBank.clear();
    for( int i = 0; i < src_gray.rows; i++ )
    {
        for( int j = 0; j < src_gray.cols; j++ )
        {
            if( myShiTomasi_dst.at<float>(i,j) > myShiTomasi_minVal + ( myShiTomasi_maxVal - myShiTomasi_minVal )*myShiTomasi_qualityLevel/max_qualityLevel )
            {
                circle( myShiTomasi_copy, Point(j,i), 4, Scalar( rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256) ), FILLED );
//                //chessboard Edge filter for second image
//                if (j < 2100 && j > 800)
//                    pointBank.push_back(Point(j,i));
                //chessboard Edge filter for second image
                if (j>200 && j<2330)
                    pointBank.push_back(Point(j,i));
            }
        }
    }
    imshow( myShiTomasi_window, myShiTomasi_copy );
}

///**
// * @function myHarris_function
// */
//void myHarris_function( int, void* )
//{
//    myHarris_copy = src.clone();
//    myHarris_qualityLevel = MAX(myHarris_qualityLevel, 1);
//
//    for( int i = 0; i < src_gray.rows; i++ )
//    {
//        for( int j = 0; j < src_gray.cols; j++ )
//        {
//            if( Mc.at<float>(i,j) > myHarris_minVal + ( myHarris_maxVal - myHarris_minVal )*myHarris_qualityLevel/max_qualityLevel )
//            {
//                circle( myHarris_copy, Point(j,i), 4, Scalar( rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256) ), FILLED );
//            }
//        }
//    }
//
//    imshow( myHarris_window, myHarris_copy );
//}

