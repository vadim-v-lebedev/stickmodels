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

#include "preprocessing.h"

//--------------------------
//--------Geometry----------
//--------------------------

double line2PointDist(Point x, vector<Point> segment){
	Point L = segment[1] - segment[0];
	//printf("%g,%g  %g\n",L.x,L.y,norm(L));
	double t = L.dot(x-segment[0])/L.dot(L);
	//printf("  %g\n",t);
	if (t < 0){
		return norm(x-segment[0]);
	}
	if (t > 1){
		return norm(x-segment[1]);
	}
	return norm(x-segment[0]-t*(segment[1]-segment[0]));
}	
double line2LineDist(vector<Point> A, vector<Point> B){
	vector<double> d(4);
	d[0] = line2PointDist(A[0],B);
	d[1] = line2PointDist(A[1],B);
	d[2] = line2PointDist(B[0],A);
	d[3] = line2PointDist(B[1],A);
	//return *min_element(d.begin(),d.end());
	//расстояние между ближайшими точками отрезков
	return min(max(d[0],d[1]),max(d[2],d[3]));
	//return 0.5*max(d[0],d[1])+0.5*max(d[2],d[3]);
	//мера того, насколько отрезки хорошо накладываются один на другой
}
double line2LineAngle(vector<Point> A, vector<Point> B){
	Point a = A[1] - A[0];
	Point b = B[1] - B[0];
	double c = a.dot(b)/norm(a)/norm(b);
	//тут случается что-то плохое, похоже вылазит погрешность вычислений
	//пришлось обрезать
	c = c > 1.0 ? 1.0 : c;
	c = c < -1.0 ? -1.0 : c;
	double phi = acos( c );
	phi = phi > M_PI/2 ? phi - M_PI : phi;
	return phi;
}
int transformLines( vector<vector<Point> >& lines, Mat M){
	for (int i = 0; i < lines.size(); i++)
		transform(lines[i], lines[i], M);
}
int copyLines( vector<vector<Point> >& lines, vector<vector<Point> >& lines2){
	lines2 = vector<vector<Point> >(lines.size());
	for (int i = 0; i < lines.size(); i++)
		lines2[i] = lines[i];
}

//--------------------
//----Visualization---
//--------------------
vector<vector<Point2i> > contoursD2I(vector<vector<Point> >& contours){
	vector<vector<Point2i> > int_contours(contours.size());
	for (int i = 0; i < contours.size(); i++){
		int_contours[i] = vector<Point2i>(2);
		int_contours[i][0] = contours[i][0];
		int_contours[i][1] = contours[i][1];
	}
	return int_contours;
}
int drawEverything(Mat& gray_background, vector<vector<Point> > model, vector<vector<Point> > lines, const string& caption){
	
	Mat M = HEIGHT/2*Mat::eye(2,3,CV_64F);
	M.at<double>(0,2) = WIDTH/2;
	M.at<double>(1,2) = HEIGHT/2;
	transformLines(lines, M);
	transformLines(model, M);
	
	Mat display;
	cvtColor(gray_background,display,CV_GRAY2BGR);
	drawContours(display, contoursD2I(lines), -1, Scalar(47,47,63), 2);
	drawContours(display, contoursD2I(model), -1, Scalar(127,191,0), 2);
	imshow(caption,display);
}


//--------------------------
//--------Cost Function-----
//--------------------------
inline double artomonoid(double x){
	return 1 - exp(-50*x);
}
double line2LineCost(vector<Point> A, vector<Point> B){
	double d = line2LineDist(A,B);
	double phi = line2LineAngle(A,B);
	//return pow(d,2)/50 + exp(phi/2) - 1;
	//printf(" d %g a %g\n",d,phi/M_PI*180);
	return artomonoid(pow(d,2) + pow(phi,2));
}
double line2SkeletonCost(vector<Point> line, vector<vector<Point> > skeleton){
	double cost = INF; 
	for (int i = 0; i < skeleton.size(); i++){
		double curr_cost = line2LineCost(line, skeleton[i]);
		if (curr_cost < cost) 
			cost = curr_cost;
	}
	return cost;
}
double skeleton2SkeletonCost(vector<vector<Point> > skeleton1, vector<vector<Point> > skeleton2){
	vector<double> costVec(skeleton1.size());
	for (int i = 0; i < skeleton1.size(); i++){
		costVec[i] = line2SkeletonCost(skeleton1[i], skeleton2);
		//printf("   %d  %g\n",i,costVec[i]);
	}
	double avg = accumulate(costVec.begin(), costVec.end(), 0.0) / costVec.size();
	return avg;
}
double transformationCost(Mat M){
	double g = 0.01; 

	double cost = 0;
	M -= Mat::eye(2,3,CV_64F);
	M.at<double>(0,2) *= g;
	M.at<double>(1,2) *= g;
	pow(M, 2, M);
	return sum(M)[0];
}
int printMatrix(Mat M){
	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 3; j++){
			printf("%3.3f ",M.at<double>(i,j));
		}
		printf("\n");
	}
}

//--------------------------
//--------Optimizer---------
//--------------------------	

int initialSearch(vector<vector<Point> >& model, vector<vector<Point> >& lines, Mat& M, double delta = 0.1){
	vector<vector<Point> > newModel;
	Mat m,bestM;
	M.copyTo(bestM);
	double currCost,bestCost = skeleton2SkeletonCost(model,lines);
	
	for (int i = -3; i <= 3; i++){
		for (int j = -3; j < 3; j++){
			M.copyTo(m);
			m.at<double>(0,2) = i*delta;
			m.at<double>(1,2) = j*delta;				
			copyLines(model,newModel);
			transformLines(newModel,m);
			currCost = skeleton2SkeletonCost(newModel,lines);
			if (currCost < bestCost){
				bestCost = currCost;
				m.copyTo(bestM);
			}
		}
	}
	
	M = bestM;
}

int optimizerStep(vector<vector<Point> >& model, vector<vector<Point> >& lines, Mat& M, Mat& grad0, double& alpha){
	vector<vector<Point> > newModel;
	copyLines(model,newModel);
	transformLines(newModel,M);
	printf("cost = %g\n",skeleton2SkeletonCost(newModel,lines));
	double f1,f2;
	Mat m,grad = Mat::zeros(2,3,CV_64F);
	
	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 3; j++){
			double delta = 1e-5;
			M.copyTo(m);
			m.at<double>(i,j) += delta;			
			copyLines(model,newModel);
			transformLines(newModel,m);
			f2 = skeleton2SkeletonCost(newModel,lines);

			M.copyTo(m);
			m.at<double>(i,j) -= delta;
			copyLines(model,newModel);
			transformLines(newModel,m);
			f1 = skeleton2SkeletonCost(newModel,lines);
			
			grad.at<double>(i,j) = (f2-f1)/(2*delta);
			
			//printf("%g %g %g : grad%d%d = %g\n",f2,f1,delta,i,j,(f2-f1)/(2*delta));
		}
	}
	
	if (sum(grad.mul(grad0))[0] < 0){
		alpha *= 0.7;
	}
	printf("alpha = %g\n",alpha);
	M = M - alpha*grad;
	printMatrix(grad);
	grad0 = grad;
	
}
	
//--------------------------
//-----------Main-----------
//--------------------------	
	
int main(int argc, char** argv)
{		
	const char* filename = argc >= 2 ? argv[1] : "car1.jpg";
	const char* modelFilename = argc >= 3 ? argv[2] : "model.jpg";

	printf("one\n");
	Mat img,modelImg,display;
	prepareImage(filename, img);
	prepareImage(modelFilename, modelImg);
	//Mat display = Mat::zeros(img.size(), CV_8UC3);
	vector<vector<Point> > lines,model,newModel;
	extractLines( img, lines);
	extractLines( modelImg, model);
	printf("%d lines in image, %d lines in model\n",lines.size(), model.size());
	
	//матрица перехода из ск связанной с углом изображения в ск связанную с его центром
	//и еще масштабирование
	Mat shiftMat = Mat::eye(2,3,CV_64F);
	shiftMat.at<double>(0,2) = -WIDTH/2;
	shiftMat.at<double>(1,2) = -HEIGHT/2;
	shiftMat /= HEIGHT/2;
	transformLines(lines, shiftMat);
	transformLines(model, shiftMat);
	//перед отображением придется домножать на обратную матрицу, но это ничего
	
	drawEverything(img, model, lines, "good");
	
	Mat M = Mat::eye(2,3,CV_64F);
	Mat grad0 = Mat::zeros(2,3,CV_64F);
	double alpha = 0.1;
	
	//initialSearch(model,lines,M);
	copyLines( model,newModel);
	transformLines( newModel, M);
	
	drawEverything(img, newModel, lines, "better");
	
	for (int k =0; k < 50; k++){
		
		optimizerStep(model, lines, M, grad0, alpha);
		copyLines( model,newModel);
		transformLines( newModel, M);
		
		drawEverything(img, newModel, lines, "the best");
		waitKey(1);
		printf("step #%d\n",k+1);
	}
	
	
	printf("\n");
	printMatrix(M);
    waitKey();
    return 0;
}

