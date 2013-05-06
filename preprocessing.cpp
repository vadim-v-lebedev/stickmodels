#include "preprocessing.h"

//--------------------------
//-------Preprocessing------
//--------------------------
int prepareImage(const char* filename, Mat& image){
	image = imread(filename, 0);
    if(image.empty()) {
        cout << "can not open " << filename << endl;
        return -1;
    }
	resize(image, image, Size(WIDTH, HEIGHT));
	return 0;
}
 
int extractLines( Mat image, vector<vector<Point> >& lines) {    

    Mat edges,smooth,c_edges;
	GaussianBlur(image, smooth, Size(0,0), 1.5);
    Canny(smooth, edges, 40, 150, 3); 
    cvtColor(edges, c_edges, CV_GRAY2BGR);

	vector<vector<Point2i> > contours;
	printf("suddenly, ");
	findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	printf("%d contours were found\n",contours.size());
	//drawContours(c_edges, contours, -1, Scalar(255,255,255));
		
	{
		const int MIN_CONTOUR_LENGTH = 30;
		vector<vector<Point2i> >::iterator it = contours.begin();
		while (it != contours.end()){
			if (it->size() < MIN_CONTOUR_LENGTH){
				contours.erase( it );
				//printf("erased\n");
			} else {
				it++;
			}
		}
	}
	
	for (int i = 0; i < contours.size(); i++) {
		//printf("%d ->",contours[i].size());
		approxPolyDP(contours[i], contours[i], 2.5, false) ;
		//printf("%d\n",contours[i].size());
	}
	
	const int MIN_LINE_LENGTH = 20;
	for (int i = 0; i < contours.size(); i++){
		for(int j = 0; j < contours[i].size()-1; j++){
			int len = hypot(contours[i][j].x - contours[i][j+1].x, contours[i][j].y - contours[i][j+1].y);
			if ( len > MIN_LINE_LENGTH) {
				vector<Point> line;
				line.push_back(contours[i][j]);
				line.push_back(contours[i][j+1]);
				//printf("  %d %d %d %d\n",contours[i][j].x,contours[i][j].y,contours[i][j+1].x,contours[i][j+1].y);
				lines.push_back(line);
			}
		}
	}
}
