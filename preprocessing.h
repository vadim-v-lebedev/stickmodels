#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "stdio.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#define INF 1e100
#define WIDTH 320
#define HEIGHT 240

#define Point Point2d

using namespace cv;
using namespace std;

//--------------------------
//-------Preprocessing------
//--------------------------

int prepareImage(const char* filename, Mat& image);
int extractLines( Mat image, vector<vector<Point> >& lines);

#endif
