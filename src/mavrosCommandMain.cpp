#include <iostream>
#include "ros/ros.h"
#include "mavrosCommand.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cstdlib>
#include <common.hpp>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace dnn;

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
    "{ device      |  0 | camera device number. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
    "{ thr         | .5 | Confidence threshold. }"
    "{ nms         | .4 | Non-maximum suppression threshold. }"
    "{ backend     |  0 | Choose one of computation backends: "
                         "0: automatically (by default), "
                         "1: Halide language (http://halide-lang.org/), "
                         "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                         "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                         "0: CPU target (by default), "
                         "1: OpenCL, "
                         "2: OpenCL fp16 (half-float precision), "
                         "3: VPU }";


float confThreshold = 0.5;
float nmsThreshold = 0.4;
std::vector<std::string> classes;

void calculateGpsPosition( int pixelX, int pixelY, double Latitude, double Longitude, double Heading, int classId, float conf);

void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

bool fileExists(const string& name);

std::vector<String> getOutputsNames(const Net& net);

int frequency = 10;

int pictureSendCount = 0;

struct GPScords{
	double Latitude;
	double Longitude;
};

bool fileExists(const string& name)
{
	if (FILE *file = fopen(name.c_str(), "r"))
	{
		fclose(file);
		return true;
	}
	
	return false;
}

int main(int argc, char** argv){

	ros::init(argc, argv, "detector");
	mavrosCommand command;
	
	ros::Rate loop_rate(frequency);
	sleep(1);
	
	CommandLineParser parser(argc, argv, keys);

    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");
    
    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run object detection deep learning networks using OpenCV.");

    //confThreshold = parser.get<float>("thr");
    //nmsThreshold = parser.get<float>("nms");
    float scale = 0.00392;
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = 416;
    int inpHeight = 416;
    std::string modelPath = "/home/maciej/Training2/yolov3-tiny-crocodile_10000.weights";
    std::string configPath = "/home/maciej/yolov3-tiny-crocodile.cfg";
    

    // Open file with classes names.
    std::string file = "/home/maciej/classes.txt";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }

    // Load a model.
    Net net = readNet(modelPath, configPath, parser.get<String>("framework"));
    net.setPreferableBackend(parser.get<int>("backend"));
    net.setPreferableTarget(parser.get<int>("target"));
	
    // Create a window
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    //namedWindow(kWinName, CV_WINDOW_AUTOSIZE);


    // Open a video file or an image file or a camera stream.
	
	Mat frame, blob;
	
	
	
	while (ros::ok()) 
	{
        if (fileExists("/home/maciej/zdj2/" + to_string(pictureSendCount) + ".jpg")
        && fileExists("/home/maciej/zdj2/" + to_string(pictureSendCount) + ".txt"))
        {
			frame = imread("/home/maciej/zdj2/" + to_string(pictureSendCount) + ".jpg");
			if (frame.empty())
			{
				waitKey();
				break;
			}

			// Create a 4D blob from a frame.
			Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
						inpHeight > 0 ? inpHeight : frame.rows);
			blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);
	
			// Run a model.
			net.setInput(blob);
			if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
			{
				resize(frame, frame, inpSize);
				Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
				net.setInput(imInfo, "im_info");
			}
			std::vector<Mat> outs;
			net.forward(outs, getOutputsNames(net));
	
			postprocess(frame, outs, net);
	
			// Put efficiency information.
			std::vector<double> layersTimes;
			double freq = getTickFrequency() / 1000;
			double t = net.getPerfProfile(layersTimes) / freq;
			std::string label = format("Inference time: %.2f ms", t);
			putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	
			//imshow(kWinName, frame);
			waitKey(1);
			pictureSendCount++;
			cout <<"Picture: " << pictureSendCount << endl;
		}
        
        
        ros::spinOnce();
		loop_rate.sleep();
    };
    return 0;
}	


void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                int left = (int)data[i + 3];
                int top = (int)data[i + 4];
                int right = (int)data[i + 5];
                int bottom = (int)data[i + 6];
                int width = right - left + 1;
                int height = bottom - top + 1;
                classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    else if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() == 1);
        float* data = (float*)outs[0].data;
        for (size_t i = 0; i < outs[0].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                int left = (int)(data[i + 3] * frame.cols);
                int top = (int)(data[i + 4] * frame.rows);
                int right = (int)(data[i + 5] * frame.cols);
                int bottom = (int)(data[i + 6] * frame.rows);
                int width = right - left + 1;
                int height = bottom - top + 1;
                classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	int centerX = round((right + left)/2) - 960;
	int centerY = (round((top + bottom)/2) - 540) * -1;
	int imageNumber = pictureSendCount;
	string txtPath = "/home/maciej/zdj2/" + to_string(imageNumber) + ".txt";
	ifstream gpsFile ;
	string line;
	gpsFile.open (txtPath);
	
	if (gpsFile.is_open())
	{
		getline(gpsFile,line);
		double Latitude = stod(line);
		cout << setprecision(10) << Latitude << endl;
		
		getline(gpsFile,line);
		double Longitude = stod(line);
		cout <<setprecision(10)<< Longitude << endl;
		
		getline(gpsFile,line);
		double Heading = stod(line);
		cout << setprecision(10)<<Heading << endl;
		
		gpsFile.close();
		
		calculateGpsPosition(centerX, centerY, Latitude, Longitude, Heading, classId, conf);
	}
	else cout << "Unable to open file";

		
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

std::vector<String> getOutputsNames(const Net& net)
{
    static std::vector<String> names;
    if (names.empty())
    {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void getLatLongShift(mavrosCommand command, double length, double angle, double &pointLatitude, double &pointLongitude)
{	
	double lat = command.toRad(pointLatitude);
	double lng = command.toRad(pointLongitude);
	
	double lat2 = asin(sin(lat) * cos((length / 1000) / 6378.1) + cos(lat) * sin((length / 1000) / 6378.1) * cos(command.toRad(angle)));
	double lng2 = lng + atan2(sin(command.toRad(angle)) * sin((length / 1000) / 6378.1) * cos(lat), cos((length / 1000) / 6378.1) - sin(lat) * sin(lat2));
	
	pointLatitude = lat2 / 3.14159265 * 180;
	pointLongitude = lng2 / 3.14159265 * 180;
}

void calculateGpsPosition( int pixelX, int pixelY, double Latitude, double Longitude, double Heading, int classId, float conf)
{
	cout << "Kalkulacja gps" << endl;
	int quarter = 0;
	mavrosCommand command;
	cout << pixelX << endl;
	cout << pixelY << endl;
	double lat = Latitude;
	double lng = Longitude;
	double distance, head;
	
	ofstream file;
	string filePath;
	if(classId == 0)
	{
		filePath = "/home/maciej/result/crocodiles.txt";
	}
	else
	{
		filePath = "/home/maciej/result/circles.txt";
	}
	file.open(filePath, ios::out | ios::app);
	
	
	if( pixelX == 0 && pixelY == 0)
	{
		head = Heading;
	}
	else if (pixelX == 0)
	{
		distance = ((double)pixelY) / 48;
		if(pixelY > 0)
		{
			head = Heading;
			getLatLongShift(command, distance, Heading, lat, lng);
		}
		else
		{
			head = fmod((Heading + 180), 360);
			getLatLongShift(command, distance, head, lat, lng);
		}
	}
	else if (pixelY == 0)
	{
		distance = ((double)pixelX) / 48;
		if(pixelX > 0)
		{
			head = fmod((Heading + 90), 360);
			getLatLongShift(command, distance, head, lat, lng);
		}
		else
		{
			head = fmod((Heading + 270), 360);
			getLatLongShift(command, distance, head, lat, lng);
		}
	}
	else if( pixelX > 0 && pixelY > 0 )
	{
		quarter = 0;
		
		distance = sqrt(pow(pixelX, 2) + pow(pixelY, 2));
		float angle = asin((sin(90) * pixelX) / distance) / 3.14159265 * 180;
		head = fmod((Heading + angle) + 360, 360);
		
		getLatLongShift(command, distance, head, lat, lng);
	}
	else if( pixelX > 0 && pixelY < 0 )
	{
		quarter = 1;
		
		distance = sqrt(pow(pixelX, 2) + pow(pixelY, 2));
		float angle = asin((sin(90) * (-pixelY)) / distance) / 3.14159265 * 180;
		head = fmod((Heading + 90 + angle) + 360, 360);
		
		getLatLongShift(command, distance, head, lat, lng);
	}
	else if( pixelX < 0 && pixelY < 0 )
	{
		quarter = 2;
		
		distance = sqrt(pow(pixelX, 2) + pow(pixelY, 2));
		float angle = asin((sin(90) * (-pixelX)) / distance) / 3.14159265 * 180;
		head = fmod((Heading + 180 + angle) + 360, 360);
		
		getLatLongShift(command, distance, head, lat, lng);
	}
	else if( pixelX < 0 && pixelY > 0 )
	{
		quarter = 3;
		distance = sqrt(pow(pixelX, 2) + pow(pixelY, 2));
		float angle = asin((sin(90) * pixelY) / distance) / 3.14159265 * 180;
		head = fmod((Heading + 270 + angle) + 360, 360);
		
		getLatLongShift(command, distance, head, lat, lng);
	}
			
	file << setprecision(10) <<lat << " " << lng << " " << conf << " " << pictureSendCount << endl;
}
