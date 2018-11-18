#ifndef MAVROS_COMMAND_HPP
#define MAVROS_COMMAND_HPP

#include "ros/ros.h"
#include "mavros_msgs/RCIn.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <pwd.h>

using namespace std;
using namespace ros;

string get_username();

class mavrosCommand {
public:
	
	mavrosCommand();
	
	virtual ~mavrosCommand();
		
	int getRCInput();
	
	double toRad(double degree);
	
private:

	void init();
	
	NodeHandle _nh;
	
	void rcInputCb(mavros_msgs::RCIn::ConstPtr msg);

	Subscriber _rcInSub;

	uint16_t _rcIn[18];
};


#endif
