// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#include <fstream>
#include <ctime>
#include <chrono>
#include "windows.h"
#include "Blob.h"

#define SHOW_STEPS            // un-comment or comment this line to show steps or not

using namespace cv;
using namespace std;

// global variables ///////////////////////////////////////////////////////////////////////////////
const Scalar SCALAR_BLACK = Scalar(0.0, 0.0, 0.0);
const Scalar SCALAR_WHITE = Scalar(255.0, 255.0, 255.0);
const Scalar SCALAR_YELLOW = Scalar(0.0, 255.0, 255.0);
const Scalar SCALAR_GREEN = Scalar(0.0, 200.0, 0.0);
const Scalar SCALAR_RED = Scalar(0.0, 0.0, 255.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(vector<Blob> &existingBlobs, vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, vector<Blob> &existingBlobs);
double distanceBetweenPoints(Point point1, Point point2);
void drawAndShowContours(Size imageSize, vector<vector<Point> > contours, string strImageName);
void drawAndShowContours(Size imageSize, vector<Blob> blobs, string strImageName);
bool checkIfBlobsCrossedTheLine(vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
void drawBlobInfoOnImage(vector<Blob> &blobs, Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCount, Mat &imgFrame2Copy);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {
	ofstream OFileObject, OFileObject1, OFileObject2;  // Create Object of Ofstream
	OFileObject.open("EventFile.txt", ios::app); // Append mode
	OFileObject1.open("LocationFile.txt", ios::app); // Append mode
	OFileObject2.open("EventCountFile.txt", ios::app); // Append mode

	
	VideoCapture capVideo;

	Mat imgFrame1;
	Mat imgFrame2;

	vector<Blob> blobs;
	Point crossingLine[2];

	int carCount = 0;
	int blobCount = 0;

	//capVideo.open("collision.mp4");
	capVideo.open("final.mp4");
	//capVideo.open("traffic.mp4");
	//capVideo.open("people.avi");
	//capVideo.open("vehicle.mp4");
	//capVideo.open("CarsDrivingUnderBridge.mp4");
	
	//capVideo.open("EventDetection.mp4");
	//capVideo.open("trafficsignal.MP4");
	

	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		cout << "error reading video file" << endl << endl;      // show error message
		_getch();                   // it may be necessary to change or remove this line if not using Windows
		return(0);                                                              // and exit program
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		cout << "error: video file must have at least two frames";
		_getch();                   // it may be necessary to change or remove this line if not using Windows
		return(0);
	}

	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	int intHorizontalLinePosition = (int)round((double)imgFrame1.rows * 0.57);

	crossingLine[0].x = 0;
	crossingLine[0].y = intHorizontalLinePosition;

	crossingLine[1].x = imgFrame1.cols - 1;
	crossingLine[1].y = intHorizontalLinePosition;

	char chCheckForEscKey = 0;

	bool blnFirstFrame = true;

	int frameCount = 2;
	long long prevmillis = 0;
	while (capVideo.isOpened() && chCheckForEscKey != 27) {

		vector<Blob> currentFrameBlobs;
		vector<Blob> allFrameBlobs;

		Mat imgFrame1Copy = imgFrame1.clone();
		Mat imgFrame2Copy = imgFrame2.clone();

		Mat imgDifference;
		Mat imgThresh;

		cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

		GaussianBlur(imgFrame1Copy, imgFrame1Copy, Size(5, 5), 0);
		GaussianBlur(imgFrame2Copy, imgFrame2Copy, Size(5, 5), 0);

		absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

		threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		imshow("imgThresh", imgThresh);

		Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat structuringElement7x7 = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat structuringElement15x15 = getStructuringElement(MORPH_RECT, Size(15, 15));

		for (unsigned int i = 0; i < 2; i++) {
			dilate(imgThresh, imgThresh, structuringElement5x5);
			dilate(imgThresh, imgThresh, structuringElement5x5);
			erode(imgThresh, imgThresh, structuringElement5x5);
		}

		Mat imgThreshCopy = imgThresh.clone();

		vector<vector<Point> > contours;

		findContours(imgThreshCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		drawAndShowContours(imgThresh.size(), contours, "imgContours");

		vector<vector<Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
			convexHull(contours[i], convexHulls[i]);
		}

		drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

		
		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);

			if (possibleBlob.currentBoundingRect.area() > 400 &&
				possibleBlob.dblCurrentAspectRatio > 0.2 &&
				possibleBlob.dblCurrentAspectRatio < 4.0 &&
				possibleBlob.currentBoundingRect.width > 30 &&
				possibleBlob.currentBoundingRect.height > 30 &&
				possibleBlob.dblCurrentDiagonalSize > 60.0 &&
				(contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
				//possibleBlob.blobId = blobCount++;
				currentFrameBlobs.push_back(possibleBlob);
				
				
				//OFileObject << possibleBlob.blobId << " " << possibleBlob.centerPositions << " " << possibleBlob.dblCurrentDiagonalSize << " "; // Writing data to file
				//OFileObject << "\n";																																//cout << "Data has been appended to file";
			}
		}
		drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

		if (blnFirstFrame == true) {
			for (auto &currentFrameBlob : currentFrameBlobs) {
				//currentFrameBlob.blobId = blobCount++;
				blobs.push_back(currentFrameBlob);
			}
		}
		else {
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
		}


		SYSTEMTIME time1;
		GetSystemTime(&time1);
		long long millis1 = (time1.wMinute * 60 * 1000) + (time1.wSecond * 1000) + time1.wMilliseconds;
		static int countTheNumberOfEvents = 0;
		static long long timeWindow = 1;

			// current date/time based on current system
			// convert now to string form
			//cout << "The local date and time is: " << dt << endl;
			// convert now to tm struct for UTC
			//tm *gmtm = gmtime(&now);
			//dt = asctime(gmtm);
			//cout << "The UTC date and time is:" << dt << endl;
		int l=1;
		int i = 1;
		SYSTEMTIME time;
		GetSystemTime(&time);
		long long millis = (time.wMinute * 60*1000)+(time.wSecond * 1000) + time.wMilliseconds;
		OFileObject << millis<<" ";
		for (auto blob : blobs) {

			vector< vector<double> > predictedLocations;
			if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
			
				int deltaX = blob.predictedNextPosition.x - blob.centerPositions.at(blob.centerPositions.size() - 1).x;
				int deltaY = blob.predictedNextPosition.y - blob.centerPositions.at(blob.centerPositions.size() - 1).y;
				OFileObject << i << " " << blob.centerPositions.at(blob.centerPositions.size()-1).x << " "
					<<blob.centerPositions.at(blob.centerPositions.size() - 1).y << " "
					<< blob.predictedNextPosition.x << " "<< blob.predictedNextPosition.y << " "
					<<(1000*abs(distanceBetweenPoints(blob.centerPositions.at(blob.centerPositions.size() - 1), blob.centerPositions.at(blob.centerPositions.size() - 2))))/((millis-prevmillis))<<" "; // Writing data to file
				
				/*predictedLocations[l].push_back(i);
				predictedLocations[l].push_back(blob.predictedNextPosition.x);
				predictedLocations[l].push_back(blob.predictedNextPosition.y);
				predictedLocations[l].push_back(blob.predictedNextPosition.x + deltaX * 2);
				predictedLocations[l].push_back(blob.predictedNextPosition.y + deltaY * 2);
				predictedLocations[l].push_back(blob.predictedNextPosition.x + deltaX * 4);
				predictedLocations[l].push_back(blob.predictedNextPosition.y + deltaY * 4);
				predictedLocations[l].push_back(blob.predictedNextPosition.x + deltaX * 6);
				predictedLocations[l].push_back(blob.predictedNextPosition.y + deltaY * 6);
				predictedLocations[l].push_back(blob.predictedNextPosition.x + deltaX * 8);
				predictedLocations[l].push_back(blob.predictedNextPosition.y + deltaY * 8);
				predictedLocations[l].push_back(blob.predictedNextPosition.x + deltaX * 10);
				predictedLocations[l].push_back(blob.predictedNextPosition.y + deltaY * 10);
				predictedLocations[l].push_back(blob.predictedNextPosition.x + deltaX * 12);
				predictedLocations[l].push_back(blob.predictedNextPosition.y + deltaY * 12);
				predictedLocations[l].push_back(blob.predictedNextPosition.x + deltaX * 14);
				predictedLocations[l].push_back(blob.predictedNextPosition.y + deltaY * 14);
				predictedLocations[l].push_back(blob.predictedNextPosition.x + deltaX * 16);
				predictedLocations[l].push_back(blob.predictedNextPosition.y + deltaY * 16);*/

				OFileObject1<<i<<" "<<blob.dblCurrentDiagonalSize<<" ";
				OFileObject1 << blob.predictedNextPosition.x << " ";
				OFileObject1 << blob.predictedNextPosition.y << " ";
				OFileObject1 << blob.predictedNextPosition.x + deltaX * 4<<" ";
				OFileObject1 << blob.predictedNextPosition.y + deltaY * 4 << " ";
				OFileObject1 << blob.predictedNextPosition.x + deltaX * 8 << " ";
				OFileObject1 << blob.predictedNextPosition.y + deltaY * 8 << " ";
				OFileObject1 << blob.predictedNextPosition.x + deltaX * 12 << " ";
				OFileObject1 << blob.predictedNextPosition.y + deltaY * 12 << " ";
				OFileObject1 << blob.predictedNextPosition.x + deltaX * 16 << " ";
				OFileObject1 << blob.predictedNextPosition.y + deltaY * 16 << " ";
				OFileObject1 << blob.predictedNextPosition.x + deltaX * 20 << " ";
				OFileObject1 << blob.predictedNextPosition.y + deltaY * 20 << " ";

				countTheNumberOfEvents++;

				SYSTEMTIME Currenttime;
				GetSystemTime(&Currenttime);
				long long currentmillis = (Currenttime.wMinute * 60 * 1000) + (Currenttime.wSecond * 1000) + Currenttime.wMilliseconds;
				if (currentmillis - millis1 >= timeWindow) {
					OFileObject2 << timeWindow << " " << countTheNumberOfEvents << endl;
					//timeWindow++;
					millis1 = currentmillis;
					countTheNumberOfEvents = 0;
				}
				
				//long int size = predictedLocations[l].size();
				
				/*OFileObject1 << i << " " << blob.centerPositions.at(blob.centerPositions.size() - 1) << " ";
				for (int counter = 0; counter < 19; counter++)
				{
					OFileObject1 << predictedLocations[l][(predictedLocations[l].size() -1)-19+counter] << " ";
				}*/
			}
			
			i++;
			//predictedLocations.clear();
		}
		prevmillis = millis;
		OFileObject<<"\n";
		OFileObject1 << endl;
		l++;
		drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

		imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

		drawBlobInfoOnImage(blobs, imgFrame2Copy);

		bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition, carCount);

		if (blnAtLeastOneBlobCrossedTheLine == true) {
			//line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
		}
		else {
			//line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
		}

		
		//drawCarCountOnImage(carCount, imgFrame2Copy); 
		//drawCarCountOnImage(carCount, imgFrame2Copy);

		imshow("imgFrame2Copy", imgFrame2Copy);

		//cv::waitKey(0);                 // uncomment this line to go frame by frame for debugging

		// now we prepare for the next iteration

		currentFrameBlobs.clear();

		imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
			capVideo.read(imgFrame2);
		}
		else {
			cout << "end of video\n";
			break;
		}

		/*string line;
		ifstream myfile("locationFile.txt");
		if (myfile.is_open())
		{
			while (getline(myfile, line))
			{
				cout << line << '\n';
			}
			myfile.close();
		}

		else cout << "Unable to open file";*/
		blnFirstFrame = false;
		frameCount++;
		chCheckForEscKey = waitKey(1);
	}


	
	
	if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
		waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows
	
	
	
	OFileObject.close(); // Closing the file
	OFileObject1.close(); // Closing the file
	OFileObject2.close(); // Closing the file
	return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {

	for (auto &existingBlob : existingBlobs) {

		existingBlob.blnCurrentMatchFoundOrNewBlob = false;

		existingBlob.predictNextPosition();
	}

	for (auto &currentFrameBlob : currentFrameBlobs) {

		int intIndexOfLeastDistance = 0;
		double dblLeastDistance = 100000.0;

		for (unsigned int i = 0; i < existingBlobs.size(); i++) {

			if (existingBlobs[i].blnStillBeingTracked == true) {

				double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

				if (dblDistance < dblLeastDistance) {
					dblLeastDistance = dblDistance;
					intIndexOfLeastDistance = i;
				}
			}
		}

		if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
		}
		else {
			addNewBlob(currentFrameBlob, existingBlobs);
		}

	}

	for (auto &existingBlob : existingBlobs) {

		if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
			existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
		}

		if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
			existingBlob.blnStillBeingTracked = false;
		}

	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

	existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
	existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

	existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

	existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
	existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

	existingBlobs[intIndex].blnStillBeingTracked = true;
	existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

	currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

	existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
	cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

	cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(Size imageSize, vector<Blob> blobs, string strImageName) {

	Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	vector<vector<Point> > contours;

	for (auto &blob : blobs) {
		if (blob.blnStillBeingTracked == true) {
			contours.push_back(blob.currentContour);
		}
	}

	drawContours(image, contours, -1, SCALAR_WHITE, -1);

	imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount) {
	bool blnAtLeastOneBlobCrossedTheLine = false;

	for (auto blob : blobs) {

		if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
			int prevFrameIndex = (int)blob.centerPositions.size() - 2;
			int currFrameIndex = (int)blob.centerPositions.size() - 1;

			if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition) {
				carCount++;
				blnAtLeastOneBlobCrossedTheLine = true;
			}

			if (blob.centerPositions[prevFrameIndex].y < intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y >= intHorizontalLinePosition) {
				carCount++;
				blnAtLeastOneBlobCrossedTheLine = true;
			}
		}

	}

	return blnAtLeastOneBlobCrossedTheLine;
}




///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {

	for (unsigned int i = 0; i < blobs.size(); i++) {

		if (blobs[i].blnStillBeingTracked == true) {
			rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

			int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
			double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
			int intFontThickness = (int)std::round(dblFontScale * 1.0);

			//cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
		}
	}
}

void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);

	//Size textSize = cv::getTextSize(carCount, intFontFace, dblFontScale, intFontThickness, 0);
	Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);


	Point ptTextBottomLeftPosition;

	ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 14);
	ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

	putText(imgFrame2Copy, "Collision Predicted", ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

}
/**/
///////////////////////////////////////////////////////////////////////////////////////////////////
/*void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

	int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
	int intFontThickness = (int)std::round(dblFontScale * 1.5);

	Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

	Point ptTextBottomLeftPosition;

	ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25)-100;
	ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

	putText(imgFrame2Copy, std::to_string(carCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

}*/











