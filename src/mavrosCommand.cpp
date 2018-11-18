#include <iostream>
#include "mavrosCommand.hpp"

using namespace std;
using namespace ros;

#define PI 3.14159265

string get_username() {
    struct passwd *pwd = getpwuid(getuid());
    if (pwd)
        return pwd->pw_name;
    else
        return "odroid";
}

mavrosCommand::mavrosCommand(){
	_nh = NodeHandle();
	init();
}

mavrosCommand::~mavrosCommand(){
}

void mavrosCommand::init(){
	_rcInSub = _nh.subscribe("mavros/rc/in", 100, &mavrosCommand::rcInputCb, this);
}

void mavrosCommand::rcInputCb(mavros_msgs::RCIn::ConstPtr msg) {
	_rcIn[5] = msg->channels[5];
}

int mavrosCommand::getRCInput(){
	return _rcIn[5];
}

double mavrosCommand::toRad(double degree){
    return degree / 180 * PI;
}
