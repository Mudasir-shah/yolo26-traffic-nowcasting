"""Evaluation videos (public, from GitHub) and their counting lines in source-pixel coordinates (y, x_min, x_max)."""
VIDEOS = {
    "urban": dict(path="data/bay_traffic.mp4", tracks="tracks_bay.json", line=(300, 0, 1280),
                  title="Urban multi-lane road", url="https://raw.githubusercontent.com/Bayunova28/Vehicle_Counting_Detection/master/Traffic.mp4",
                  source="github.com/Bayunova28/Vehicle_Counting_Detection"),
    "freeway": dict(path="data/bridge.mp4", tracks="tracks_bridge.json", line=(150, 0, 1280),
                    title="Freeway seen from a bridge", url="https://raw.githubusercontent.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp/master/CarsDrivingUnderBridge.mp4",
                    source="github.com/MicrocontrollersAndMore/OpenCV_3_Car_Counting_Cpp"),
    "residential": dict(path="data/az_input.mp4", tracks="tracks.json", line=(232, 196, 640),
                        title="Residential street", url="https://raw.githubusercontent.com/ahmetozlu/vehicle_counting_tensorflow/master/input_video.mp4",
                        source="github.com/ahmetozlu/vehicle_counting_tensorflow"),
}
FREEWAY_LINE_SWEEP = [110, 150, 190, 250]
