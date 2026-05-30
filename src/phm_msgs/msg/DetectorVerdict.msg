std_msgs/Header header
string source
float32 score            # normalized severity in [0,1], 0 healthy, 1 worst
bool violating           # post-hysteresis boolean
string reason
uint8 suggested_action
