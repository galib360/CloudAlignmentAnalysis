#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <jsoncpp/json/json.h>

using namespace cv;
using namespace std;

//typedef struct {
//	vector<vector<Point2f>> pnts2d;
//} campnts;
//vector<campnts> pnts(NUMBER_OF_POINTS,NUMBER_OF_POINTS);

int thresh = 220;
int max_thresh = 255;
static int howmanycam = 3;
static int howmanyframe = 2;
static int howmanypc = 4;
int NUMBER_OF_POINTS = 0;
static int savepoints = 0; //0 for reading points from file
static int quatmethod = 1; //1 for angle calc; 0 for document method
static int readjson = 1;// value set to 1 for read cam param from json files; 0 for txt file
ifstream readfile;
ofstream myfile;
string outputfilename = "points.txt";
string inputdir = "data1904v2/cam0";

//vector<Mat> srcs;
//vector<Mat> grays;
//vector<Mat> dsts;
vector<vector<Mat>> P(howmanycam);
vector<vector<Mat>> Ks(howmanycam);
vector<vector<Mat>> Rs(howmanycam);
vector<vector<Mat>> Rts(howmanycam);
vector<vector<Mat>> ts(howmanycam);
vector<vector<Mat>> quats(howmanycam);
vector<vector<Mat>> Cs(howmanycam);

vector<vector<vector<Point2f>>> pnts2d(howmanycam,
		vector<vector<Point2f> >(howmanyframe));
//vector<vector<vector<int> > > vec (5,vector<vector<int> >(3,vector <int>(2,4)));

//Point2f P2d [4][2][10];

vector<vector<Mat>> points3D(howmanycam);
vector<vector<Mat>> points3Dnorm(howmanycam);

vector<vector<Point3f>> P3D(howmanycam);

vector<Mat> RsPW(howmanypc);
vector<Mat> tsPW(howmanypc);

vector<Mat> RsNE(howmanypc);
vector<Mat> tsNE(howmanypc);
vector<Mat> quatsNE(howmanypc);

float ComputeDistance(float x1, float y1, float z1, float x2, float y2,
		float z2) {

	float diffX = x1 - x2;
	float diffY = y1 - y2;
	float diffZ = z1 - z2;
	float distance = sqrt((diffX * diffX) + (diffY * diffY) + (diffZ * diffZ));

	return distance;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
	vector<int> config = *((vector<int>*) userdata);
	//*((int*)userdata);
	int whichcam = config[0];
	int whichframe = config[1];
	if (event == EVENT_LBUTTONDOWN) {
		cout << "Left button of the mouse is clicked - position (" << x << ", "
				<< y << ") in " << whichcam << endl;
//		Point2f point = ((Point_<float> ) x, (Point_<float> ) y);
		Point2f point(x, y);
		cout << "point is " << point << endl;

		pnts2d[whichcam][whichframe].push_back(point);
		myfile << whichcam << " " << whichframe << " " << x << " " << y << endl;
		if (whichcam == 0 && whichframe == 0) {
			NUMBER_OF_POINTS++;
		}

//		P2d [4][2][10] = point;
	}
}

void cornerHarris_demo(Mat &gray, int whichcam, int whichframe) {
	vector<int> config(2);
	config[0] = whichcam;
	config[1] = whichframe;
	cout << "called harris demo" << endl;
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	vector<Point2f> corners;
	Mat dst = Mat::zeros(gray.size(), CV_32FC1);
	cornerHarris(gray, dst, blockSize, apertureSize, k);
	Mat dst_norm, dst_norm_scaled;
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
//	dsts.push_back(dst_norm_scaled);
	int countCorners = 0;
	for (int i = 0; i < dst_norm.rows; i++) {
		for (int j = 0; j < dst_norm.cols; j++) {
			if ((int) dst_norm.at<float>(i, j) > thresh) {
				//circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
				corners.push_back(Point(j, i));
				//cout<<Point(j,i)<<endl;
				countCorners++;
			}
		}
	}
	cout << "total corners found : " << countCorners << endl;

	namedWindow("ImageDisplay", WINDOW_NORMAL);
	setMouseCallback("ImageDisplay", CallBackFunc, &config);
	imshow("ImageDisplay", dst_norm_scaled);
//    imwrite("corners.png", dst_norm_scaled);

	waitKey(0);
//	cout<<"Points written"<<endl;

//    cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
//			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

}

int main() {

	if (savepoints == 1) {
		myfile.open(outputfilename);
	}

//Result from pairwise alignment stored---->
//	RsPW[0] = Mat::eye(3, 3, CV_32F);
//	float dataR1[9] = { 0.999991, -0.000720522, 0.00410299, 0.000692502,
//			0.999976, 0.00682649, -0.00410781, -0.00682359, 0.999968 };
//	RsPW[1] = cv::Mat(3, 3, CV_32F, dataR1);
//	float dataR2[9] = { 0.999947, -0.000934231, 0.0102228, 0.000893673,
//			0.999992, 0.00397126, -0.0102264, -0.00396192, 0.99994 };
//	RsPW[2] = cv::Mat(3, 3, CV_32F, dataR2);
//	float dataR3[9] = { 0.998787, -0.00431866, -0.0490552, 0.00572076, 0.999578,
//			0.0284779, 0.0489115, -0.028724, 0.99839 };
//	RsPW[3] = cv::Mat(3, 3, CV_32F, dataR3);
//
//	tsPW[0] = Mat::zeros(3, 1, CV_32F);
//	float dataT1[3] = { 0.00906843, 0.0256253, 0.0444551 };
//	tsPW[1] = cv::Mat(3, 1, CV_32F, dataT1);
//	float dataT2[3] = { 0.028302, 0.0226117, 0.0116035 };
//	tsPW[2] = cv::Mat(3, 1, CV_32F, dataT2);
//	float dataT3[3] = { 0.0447484, 0.0151607, 0.0750883 };
//	tsPW[3] = cv::Mat(3, 1, CV_32F, dataT3);

	float dataR1[9] = { 0.992237, 0.00557466, 0.124237, -0.00454524, 0.999953,
			-0.00856787, -0.124279, 0.00793667, 0.992216 };

	float dataR2[9] = { 0.999373, 0.00450517, 0.0351071, -0.00517983, 0.999803,
			0.0191499, -0.0350139, -0.0193197, 0.9992 };

	float dataR3[9] = { 0.999502, -0.00188404, 0.0314827, 0.00214578, 0.999964,
			-0.00828202, -0.031466, 0.00834546, 0.99947 };

	RsPW[0] = Mat::eye(3, 3, CV_32F);
	RsPW[1] = cv::Mat(3, 3, CV_32F, dataR1);
	RsPW[2] = cv::Mat(3, 3, CV_32F, dataR3);
	RsPW[3] = cv::Mat(3, 3, CV_32F, dataR3);

	float dataT1[3] = { -0.127405, 0.094442, 0.0917713 };

	float dataT2[3] = { 0.0655498, 0.040169, 0.126631 };

	float dataT3[3] = { -0.0568461, 0.0256409, 0.0848358 };

	tsPW[0] = Mat::zeros(3, 1, CV_32F);
	tsPW[1] = cv::Mat(3, 1, CV_32F, dataT1);
	tsPW[2] = cv::Mat(3, 1, CV_32F, dataT3);
	tsPW[3] = cv::Mat(3, 1, CV_32F, dataT3);

	//Read transformation wrt anchor----------------------->
	RsNE[0] = Mat::eye(3, 3, CV_32F);
	tsNE[0] = Mat::zeros(3, 1, CV_32F);
	quatsNE[0] = Mat::zeros(4, 1, CV_32F);

	for (int cam = 1; cam < howmanycam; cam++) {
		ifstream jsoninputNE(
				inputdir + to_string(cam) + "/" + "transformation.json");
		Json::Reader reader;
		Json::Value object;
		reader.parse(jsoninputNE, object);
		RsNE[cam] = cv::Mat::ones(3, 3, CV_32F);
		tsNE[cam] = cv::Mat::ones(3, 1, CV_32F);

		quatsNE[cam] = Mat::zeros(4, 1, CV_32F);

		tsNE[cam].at<float>(0,0) = object["position"]["x"].asFloat();
		tsNE[cam].at<float>(1,0) = object["position"]["y"].asFloat();
		tsNE[cam].at<float>(2,0) = object["position"]["z"].asFloat();

		float qw = quatsNE[cam].at<float>(0, 0) = object["rotation"]["w"].asFloat();
		float qx = quatsNE[cam].at<float>(1, 0) = object["rotation"]["x"].asFloat();
		float qy = quatsNE[cam].at<float>(2, 0) = object["rotation"]["y"].asFloat();
		float qz = quatsNE[cam].at<float>(3, 0) = object["rotation"]["z"].asFloat();

		RsNE[cam].at<float>(0, 0) = 2 * ((qw * qw) + (qx * qx) - 0.5);
		RsNE[cam].at<float>(0, 1) = 2 * ((qx * qy) - (qw * qz));
		RsNE[cam].at<float>(0, 2) = 2 * ((qw * qy) + (qx * qz));
		RsNE[cam].at<float>(1, 0) = 2 * ((qw * qz) + (qx * qy));
		RsNE[cam].at<float>(1, 1) = 2 * ((qw * qw) + (qy * qy) - 0.5);
		RsNE[cam].at<float>(1, 2) = 2 * ((qy * qz) - (qw * qx));
		RsNE[cam].at<float>(2, 0) = 2 * ((qx * qz) - (qw * qy));
		RsNE[cam].at<float>(2, 1) = 2 * ((qw * qx) + (qy * qz));
		RsNE[cam].at<float>(2, 2) = 2 * ((qy * qy) + (qz * qz) - 0.5);

	}




	//Main ---------------------->



	for (int cam = 0; cam < howmanycam; cam++) {
		for (int frame = 0; frame < 2; frame++) {
			string inputfile = inputdir + to_string(cam) + "/"
					+ to_string(frame) + ".jpg";
			Mat img = imread(inputfile);
			Mat gray;
			cvtColor(img, gray, CV_BGR2GRAY);
//			srcs.push_back(img);
//			grays.push_back(gray);

//////////read camera matrices---------->
			//for json
			if (readjson == 1) {
				ifstream jsoninput(
						inputdir + to_string(cam) + "/" + to_string(frame)
								+ ".json");
				Json::Reader reader;
				Json::Value object;
				reader.parse(jsoninput, object);
				Mat kk(3, 3, cv::DataType<float>::type, Scalar(0));
				Mat rotm(3, 3, cv::DataType<float>::type, Scalar(1));
				Mat Rt(3, 4, cv::DataType<float>::type, Scalar(1));
				Mat tvec(3, 1, cv::DataType<float>::type, Scalar(1));
				Mat quat(4, 1, cv::DataType<float>::type, Scalar(1));

				kk.at<float>(0, 0) =
						object["camera"]["focalLength"]["x"].asFloat();
				kk.at<float>(0, 1) = 0;
				kk.at<float>(0, 2) =
						object["camera"]["principalPoint"]["x"].asFloat();
				kk.at<float>(1, 0) = 0;
				kk.at<float>(1, 1) =
						object["camera"]["focalLength"]["y"].asFloat();
				kk.at<float>(1, 2) =
						object["camera"]["principalPoint"]["y"].asFloat();
				kk.at<float>(2, 0) = 0;
				kk.at<float>(2, 1) = 0;
				kk.at<float>(2, 2) = 1;

				//push kk
				Ks[cam].push_back(kk);


				float qw = quat.at<float>(0, 0) =
						object["camera"]["rotation"]["w"].asFloat();
				float qx = quat.at<float>(0, 1) =
						object["camera"]["rotation"]["x"].asFloat();
				float qy = quat.at<float>(0, 2) =
						object["camera"]["rotation"]["y"].asFloat();
				float qz = quat.at<float>(0, 3) =
						object["camera"]["rotation"]["z"].asFloat();

				qx = quat.at<float>(1, 0) *= -1;
				//qy = quat.at<float>(2, 0) *= -1;
				qz = quat.at<float>(3, 0) *= -1;

				/////////////////////////////
				quats[cam].push_back(quat);

				if (quatmethod == 1) {

					Mat rvec(3, 1, cv::DataType<float>::type, Scalar(1));
					float angle = 2 * acos(qw);
					rvec.at<float>(0, 0) = (qx / sqrt(1 - qw * qw)) * angle;
					rvec.at<float>(1, 0) = (qy / sqrt(1 - qw * qw)) * angle;
					rvec.at<float>(2, 0) = (qz / sqrt(1 - qw * qw)) * angle;
					Rodrigues(rvec, rotm);
				}

				if (quatmethod == 0) {
					rotm.at<float>(0, 0) = 2 * ((qw * qw) + (qx * qx) - 0.5);
					rotm.at<float>(0, 1) = 2 * ((qx * qy) - (qw * qz));
					rotm.at<float>(0, 2) = 2 * ((qw * qy) + (qx * qz));
					rotm.at<float>(1, 0) = 2 * ((qw * qz) + (qx * qy));
					rotm.at<float>(1, 1) = 2 * ((qw * qw) + (qy * qy) - 0.5);
					rotm.at<float>(1, 2) = 2 * ((qy * qz) - (qw * qx));
					rotm.at<float>(2, 0) = 2 * ((qx * qz) - (qw * qy));
					rotm.at<float>(2, 1) = 2 * ((qw * qx) + (qy * qz));
					rotm.at<float>(2, 2) = 2 * ((qy * qy) + (qz * qz) - 0.5);

				}

				rotm = rotm.t();
				Rs[cam].push_back(rotm);


				tvec.at<float>(0, 0) =
						object["camera"]["position"]["x"].asFloat();
				tvec.at<float>(1, 0) =
						object["camera"]["position"]["y"].asFloat();
				tvec.at<float>(2, 0) =
						object["camera"]["position"]["z"].asFloat();

				//push tvec to Cs
				Cs[cam].push_back(tvec);


				tvec = -rotm * tvec;


				//push tvec to ts
				ts[cam].push_back(tvec);

				Rt.at<float>(0, 0) = rotm.at<float>(0, 0);
				Rt.at<float>(0, 1) = rotm.at<float>(0, 1);
				Rt.at<float>(0, 2) = rotm.at<float>(0, 2);
				Rt.at<float>(1, 0) = rotm.at<float>(1, 0);
				Rt.at<float>(1, 1) = rotm.at<float>(1, 1);
				Rt.at<float>(1, 2) = rotm.at<float>(1, 2);
				Rt.at<float>(2, 0) = rotm.at<float>(2, 0);
				Rt.at<float>(2, 1) = rotm.at<float>(2, 1);
				Rt.at<float>(2, 2) = rotm.at<float>(2, 2);
				Rt.at<float>(0, 3) = tvec.at<float>(0, 0);
				Rt.at<float>(1, 3) = tvec.at<float>(1, 0);
				Rt.at<float>(2, 3) = tvec.at<float>(2, 0);

				Rts[cam].push_back(Rt);

				Mat Ptemp = kk * Rt;
				P[cam].push_back(Ptemp);

				//cout<<"Projection Matrix: "<< P[i]<<endl;

				if (savepoints == 1) {
					cornerHarris_demo(gray, cam, frame);
				}

				else if (savepoints == 0 && cam == 0 && frame == 0) {//read points from file
					readfile = ifstream(outputfilename);
					//					vector<string> fid;
					vector<string> line2;
					std::string singleline;

					int c2 = 0;

					while (std::getline(readfile, singleline)) {
						line2.push_back(singleline);
						//						cout << singleline << endl;

						stringstream linestream2(singleline);
						string val2;
						vector<string> linedata2;
						while (linestream2 >> val2) {
							linedata2.push_back(val2);
							//cout<<val<<endl;
						}

						c2 = 0;
						while (c2 < linedata2.size()) {
							//							cout<<linedata2.size()<<endl;
							int camera = strtof((linedata2[c2]).c_str(), 0);
							c2++;
							int frame = strtof((linedata2[c2]).c_str(), 0);
							c2++;
							float x = strtof((linedata2[c2]).c_str(), 0);
							c2++;
							float y = strtof((linedata2[c2]).c_str(), 0);
							c2++;
							Point2f temp(x, y);
							//							cout << "point is for " << camera << ", " << frame
							//									<< temp << endl;

							pnts2d[camera][frame].push_back(temp);
							if (camera == 0 && frame == 0) {
								NUMBER_OF_POINTS++;
							}
						}

					}

				}				//else end

			}

			// for .txt/////////////////////////////////////////////////////////////////////////////////

			if (readjson == 0) {

				ifstream txtfile = ifstream(
						inputdir + to_string(cam) + "/" + to_string(frame)
								+ ".txt");
				vector<string> fid;
				std::string line;
				vector<string> linedata;
				int c = 0;

				while (std::getline(txtfile, line)) {
					std::stringstream linestream(line);
					string val;
					while (linestream >> val) {
						linedata.push_back(val);
						//cout<<val<<endl;
					}
				}

				while (c < linedata.size()) {
					fid.push_back(linedata[c]);
					c++;
					//Put data into K
					Mat kk(3, 3, cv::DataType<float>::type, Scalar(1));
					Mat rotm(3, 3, cv::DataType<float>::type, Scalar(1));
					Mat Rt(3, 4, cv::DataType<float>::type, Scalar(1));
					Mat tvec(3, 1, cv::DataType<float>::type, Scalar(1));
					Mat quat(4, 1, cv::DataType<float>::type, Scalar(1));
					for (int j = 0; j < 3; j++) {
						for (int k = 0; k < 3; k++) {
							float temp = strtof((linedata[c]).c_str(), 0);

							kk.at<float>(j, k) = temp;
							c++;
						}
					}
					//kk = kk.t();
					Ks[cam].push_back(kk);

					for (int j = 0; j < 4; j++) {
						float temp = strtof((linedata[c]).c_str(), 0);
						quat.at<float>(j, 0) = temp;
						c++;
					}
					//for Mirror along y axis

					quat.at<float>(1, 0) *= -1;
					//				quat.at<float>(2, 0) *= -1;
					quat.at<float>(3, 0) *= -1;

					/////////////////////////////
					quats[cam].push_back(quat);

					//cout<<"quat : " << quat<<endl;

					for (int j = 0; j < 3; j++) {
						float temp = strtof((linedata[c]).c_str(), 0);
						tvec.at<float>(j, 0) = temp;
						c++;
					}
					Cs[cam].push_back(tvec);

					float qw = quat.at<float>(0, 0);
					float qx = quat.at<float>(1, 0);
					float qy = quat.at<float>(2, 0);
					float qz = quat.at<float>(3, 0);

					if (quatmethod == 1) {

						Mat rvec(3, 1, cv::DataType<float>::type, Scalar(1));
						float angle = 2 * acos(qw);
						rvec.at<float>(0, 0) = (qx / sqrt(1 - qw * qw)) * angle;
						rvec.at<float>(1, 0) = (qy / sqrt(1 - qw * qw)) * angle;
						rvec.at<float>(2, 0) = (qz / sqrt(1 - qw * qw)) * angle;
						Rodrigues(rvec, rotm);
					}

					if (quatmethod == 0) {
						rotm.at<float>(0, 0) = 2
								* ((qw * qw) + (qx * qx) - 0.5);
						rotm.at<float>(0, 1) = 2 * ((qx * qy) - (qw * qz));
						rotm.at<float>(0, 2) = 2 * ((qw * qy) + (qx * qz));
						rotm.at<float>(1, 0) = 2 * ((qw * qz) + (qx * qy));
						rotm.at<float>(1, 1) = 2
								* ((qw * qw) + (qy * qy) - 0.5);
						rotm.at<float>(1, 2) = 2 * ((qy * qz) - (qw * qx));
						rotm.at<float>(2, 0) = 2 * ((qx * qz) - (qw * qy));
						rotm.at<float>(2, 1) = 2 * ((qw * qx) + (qy * qz));
						rotm.at<float>(2, 2) = 2
								* ((qy * qy) + (qz * qz) - 0.5);
//						rotm = rotm.t();
					}

					rotm = rotm.t();
					Rs[cam].push_back(rotm);

					tvec = -rotm * tvec;
					ts[cam].push_back(tvec);

					Rt.at<float>(0, 0) = rotm.at<float>(0, 0);
					Rt.at<float>(0, 1) = rotm.at<float>(0, 1);
					Rt.at<float>(0, 2) = rotm.at<float>(0, 2);
					Rt.at<float>(1, 0) = rotm.at<float>(1, 0);
					Rt.at<float>(1, 1) = rotm.at<float>(1, 1);
					Rt.at<float>(1, 2) = rotm.at<float>(1, 2);
					Rt.at<float>(2, 0) = rotm.at<float>(2, 0);
					Rt.at<float>(2, 1) = rotm.at<float>(2, 1);
					Rt.at<float>(2, 2) = rotm.at<float>(2, 2);
					Rt.at<float>(0, 3) = tvec.at<float>(0, 0);
					Rt.at<float>(1, 3) = tvec.at<float>(1, 0);
					Rt.at<float>(2, 3) = tvec.at<float>(2, 0);

					Rts[cam].push_back(Rt);

					Mat Ptemp = kk * Rt;
					P[cam].push_back(Ptemp);

					//cout<<"Projection Matrix: "<< P[i]<<endl;

					if (savepoints == 1) {
						cornerHarris_demo(gray, cam, frame);
					}

					else if (savepoints == 0 && cam == 0 && frame == 0) {//read points from file
						readfile = ifstream(outputfilename);
						//					vector<string> fid;
						vector<string> line2;
						std::string singleline;

						int c2 = 0;

						while (std::getline(readfile, singleline)) {
							line2.push_back(singleline);
							//						cout << singleline << endl;

							stringstream linestream2(singleline);
							string val2;
							vector<string> linedata2;
							while (linestream2 >> val2) {
								linedata2.push_back(val2);
								//cout<<val<<endl;
							}

							c2 = 0;
							while (c2 < linedata2.size()) {
								//							cout<<linedata2.size()<<endl;
								int camera = strtof((linedata2[c2]).c_str(), 0);
								c2++;
								int frame = strtof((linedata2[c2]).c_str(), 0);
								c2++;
								float x = strtof((linedata2[c2]).c_str(), 0);
								c2++;
								float y = strtof((linedata2[c2]).c_str(), 0);
								c2++;
								Point2f temp(x, y);
								//							cout << "point is for " << camera << ", " << frame
								//									<< temp << endl;

								pnts2d[camera][frame].push_back(temp);
								if (camera == 0 && frame == 0) {
									NUMBER_OF_POINTS++;
								}
							}

						}

					}				//else end

				}
			}

		}
	}

	for (int cam = 0; cam < howmanycam; cam++) {
		Mat temp(4, pnts2d[cam][0].size(), CV_32F);
		triangulatePoints(P[cam][0], P[cam][1], pnts2d[cam][0], pnts2d[cam][1],
				temp);
		points3D[cam].push_back(temp);

//		cout<<"For cam"<<cam<<" the 3D points are: "<<endl;
//		cout<<temp<<endl;

		for (int k = 0; k < temp.cols; k++) {
			Point3f temporary;
			for (int j = 0; j < 4; j++) {
				temp.at<float>(j, k) = temp.at<float>(j, k)
						/ temp.at<float>(3, k);
				if (j == 0) {
					temporary.x = temp.at<float>(j, k);
				}
				if (j == 1) {
					temporary.y = temp.at<float>(j, k);
				}
				if (j == 2) {
					temporary.z = temp.at<float>(j, k);
				}
			}
			P3D[cam].push_back(temporary);
			cout << temporary << endl;
		}
		points3Dnorm[cam].push_back(temp);

//		Mat tempnormal;
//		convertPointsFromHomogeneous(temp, tempnormal);
//		cout<<tempnormal<<endl;
//		tempnormal = tempnormal.t();
//		cout<<tempnormal<<endl;
//		for (int k = 0; k < tempnormal.cols; k++) {
//			Vec3f temporary;
//			for (int j = 0; j < 3; j++) {
//
//				temporary[j] = tempnormal.at<float>(j, k);
//
//			}
//			P3D[cam].push_back(temporary);
//			cout << temporary << endl;
//		}
//		points3Dnorm[cam].push_back(tempnormal);
//		cout<<"For cam"<<cam<<" the 3D points are: "<<endl;
//		cout<<temp<<endl;
	}

	int count = 0;
	float finaldistanceMU = 0;

	//for(int cam = 0; cam<4; cam++){
	for (int aux = 1; aux < howmanycam; aux++) {
		for (int p = 0; p < NUMBER_OF_POINTS; p++) {
			float x1 = P3D[0][p].x;
			float y1 = P3D[0][p].y;
			float z1 = P3D[0][p].z;
			float x2 = P3D[aux][p].x;
			float y2 = P3D[aux][p].y;
			float z2 = P3D[aux][p].z;
			//float distance = hypot(hypot(x1 - x2, y1 - y2), z1 - z2);
			float distance = ComputeDistance(x1, y1, z1, x2, y2, z2);
			finaldistanceMU += distance;
			//if(cam!=aux){
			count++;
			//}

//			cout << distance << endl;
		}

	}
	//}
	finaldistanceMU = finaldistanceMU / count;
	cout << count << endl;
	cout << "Average distance in Mu: " << finaldistanceMU << endl;

//	Rs[1][1].convertTo(Rs[1][1], CV_64F);
//	ts[1][1].convertTo(ts[1][1], CV_64F);
//	Ks[1][1].convertTo(Ks[1][1], CV_64F);
//	std::vector<cv::Point2f> image_pointsU;
//
//	projectPoints(P3D[1], Rs[1][1], ts[1][1], Ks[1][1], cv::noArray(),
//			image_pointsU);
//	cout << image_pointsU << endl;

//////////----------------Apply PW transformation
	cout << "3D points after PW transformation : " << endl;
	for (int cam = 0; cam < howmanycam; cam++) {
		for (int p = 0; p < NUMBER_OF_POINTS; p++) {
			Mat temp3D(3, 1, cv::DataType<float>::type, Scalar(1));
			temp3D.at<float>(0, 0) = P3D[cam][p].x;
			temp3D.at<float>(1, 0) = P3D[cam][p].y;
			temp3D.at<float>(2, 0) = P3D[cam][p].z;

			temp3D = (RsPW[cam] * temp3D) + tsPW[cam];
//			temp3D = (RsPW[cam].t() * temp3D) - tsPW[cam];

			P3D[cam][p].x = temp3D.at<float>(0, 0);
			P3D[cam][p].y = temp3D.at<float>(1, 0);
			P3D[cam][p].z = temp3D.at<float>(2, 0);

			cout << P3D[cam][p] << endl;
		}

	}

	/////-------------compute distance again
	count = 0;
	float finaldistanceMPW = 0;

	//for(int cam = 0; cam<4; cam++){
	for (int aux = 1; aux < howmanycam; aux++) {
		for (int p = 0; p < NUMBER_OF_POINTS; p++) {
			float x1 = P3D[0][p].x;
			float y1 = P3D[0][p].y;
			float z1 = P3D[0][p].z;
			float x2 = P3D[aux][p].x;
			float y2 = P3D[aux][p].y;
			float z2 = P3D[aux][p].z;
			//float distance = hypot(hypot(x1 - x2, y1 - y2), z1 - z2);
			float distance = ComputeDistance(x1, y1, z1, x2, y2, z2);
			finaldistanceMPW += distance;
			//if(cam!=aux){
			count++;
			//}

			//			cout << distance << endl;
		}

	}
	//}
	finaldistanceMPW = finaldistanceMPW / count;
	cout << count << endl;
	cout << "Average distance in MPW: " << finaldistanceMPW << endl;

	if (savepoints == 1) {
		myfile.close();
	}

	cout << "Total Number of points in one frame : " << NUMBER_OF_POINTS
			<< endl;

//	Mat fun1to0 = findFundamentalMat(pnts2d[0][0],pnts2d[1][0],CV_FM_RANSAC);
//	cout<<fun1to0<<endl;
//
//	Mat imagepoints;
//	Point3f object_point_x(1, 0, 0);
//	Mat point3(0.639706, 2.96422, 0.414564);

//	std::vector<cv::Point3f> object_points;
//	cv::Point3f object_point_x(1, 0, 0);
//	object_points.push_back(object_point_x);
//	std::vector<cv::Point2f> image_pointsPW;
//
//	Rs[0][0].convertTo(Rs[0][0], CV_64F);
//	RsPW[0].convertTo(RsPW[0], CV_64F);
//	Mat R = Rs[0][0] * RsPW[0];
//
//	ts[0][0].convertTo(ts[0][0], CV_64F);
//	tsPW[0].convertTo(tsPW[0], CV_64F);
//	Mat t = ts[0][0] + tsPW[0];
//	Ks[0][0].convertTo(Ks[0][0], CV_64F);
//
//	projectPoints(P3D[0], R , t, Ks[0][0], cv::noArray(), image_pointsPW);
//	cout<<image_pointsPW<<endl;

	//Read cropped XYZs here-------->

	vector<vector<Point3f>> XYZ(howmanycam);


	for (int i = 0; i < howmanycam; i++) {
		ifstream txtfile = ifstream("xyz/" + to_string(i) + ".xyz");

		std::string line;
		vector<string> linedata;


		while (std::getline(txtfile, line)) {
			std::stringstream linestream(line);
			string val;
			int c = 0;
			Point3f temp;
			while (linestream >> val) {
				linedata.push_back(val);
				while (c < linedata.size()) {
					if (c == 0)
						temp.x = strtof((linedata[c]).c_str(), 0);
					if (c == 1)
						temp.y = strtof((linedata[c]).c_str(), 0);
					if (c == 2)
						temp.z = strtof((linedata[c]).c_str(), 0);

					c++;
				}
				//cout<<val<<endl;
			}
			XYZ[i].push_back(temp);
		}


	}

	for(int i = 0; i<howmanycam; i++){
		Rs[i][1].convertTo(Rs[i][1], CV_64F);
		ts[i][1].convertTo(ts[i][1], CV_64F);
		Ks[i][1].convertTo(Ks[i][1], CV_64F);
		std::vector<cv::Point2f> image_points;

		projectPoints(XYZ[i], Rs[i][1], ts[i][1], Ks[i][1], cv::noArray(),
				image_points);

		float xmax = -1;
		float xmin = 2000;
		float ymax = -1;
		float ymin = 2000;
		for (int j = 0; j < image_points.size(); j++) {
			if (image_points[j].x < xmin) {
				xmin = image_points[j].x;
			}
			if (image_points[j].x > xmax) {
				xmax = image_points[j].x;
			}
			if (image_points[j].y < ymin) {
				ymin = image_points[j].y;
			}
			if (image_points[j].y > ymax) {
				ymax = image_points[j].y;
			}
		}
		cout<<"for camera "<< to_string(i)<< " the min pixel values are(x,y) : "<<xmin<<", "<<ymin<<endl;
		cout<<"for camera "<< to_string(i)<< " the max pixel values are(x,y) : "<<xmax<<", "<<ymax<<endl;
		cout<<" "<<endl;
	}


	return 0;
}
