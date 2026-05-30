std_msgs/Header header
float32[] embedding      # flattened policy internal vector (e.g. 512-D recurrent feature)
uint32 dim               # len(embedding), for validation
string policy_id         # which policy produced it, e.g. "phoenix" | "smolvla"
