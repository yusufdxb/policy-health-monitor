std_msgs/Header header
uint8 state              # 0 OK, 1 DEGRADED, 2 INTERVENE, 3 STOP
uint8 STATE_OK=0
uint8 STATE_DEGRADED=1
uint8 STATE_INTERVENE=2
uint8 STATE_STOP=3
float32 score            # worst normalized severity score in [0,1] across sources
string reason            # human-readable, e.g. "ood: rolling-spread 0.012 < thr 0.019 for 3 frames"
string source            # which detector drove the state, e.g. "phm_ood" | "freq:/scan" | "threshold:cpu"
uint8 suggested_action   # 0 NONE, 1 LOG_ONLY, 2 HOLD, 3 STOP_AND_HOLD, 4 REWIND
uint8 ACTION_NONE=0
uint8 ACTION_LOG_ONLY=1
uint8 ACTION_HOLD=2
uint8 ACTION_STOP_AND_HOLD=3
uint8 ACTION_REWIND=4
