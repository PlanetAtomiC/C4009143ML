import logging

#Dataset loaded here
dataset_path = r'dft-road-casualty-statistics-collision-1979-latest-published-year.csv'
Sheff_code =  215

Lat_min = 53.3
Lat_max = 53.5
Lon_min = -1.8
Lon_max = -1.3
Map_centre = [53.383, -1.47]

#For model training

Rand_state = 42
test_size = 0.4
val_size = 0.5
kfold_splits = 5
Map_sample_size = 2000

#Features
class_features = ['weather_conditions', 'road_type', 'light_conditions', 'speed_limit', 'number_of_vehicles', 'road_surface_conditions', 'junction_detail', 'junction_control', 'urban_or_rural_area', 'day_of_week', 'hour', 'month', 'pedestrian_crossing', 'carriageway_hazards', 'special_conditions_at_site', 'did_police_officer_attend_scene_of_accident', 'is_rush_hour', 'is_weekend', 'is_dark', 'is_bad_weather']

Reg_features = ['weather_conditions', 'road_type', 'light_conditions', 'speed_limit', 'number_of_vehicles', 'road_surface_conditions', 'junction_detail', 'junction_control', 'urban_or_rural_area', 'day_of_week', 'hour', 'month', 'collision_severity', 'pedestrian_crossing', 'carriageway_hazards', 'special_conditions_at_site', 'first_road_class', 'second_road_class', 'is_rush_hour', 'is_weekend', 'is_bad_weather']

Cluster_features = ['latitude', 'longitude', 'hour', 'month', 'day_of_week', 'speed_limit', 'road_type', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'number_of_vehicles', 'number_of_casualties', 'urban_or_rural_area', 'junction_detail', 'carriageway_hazards', 'pedestrian_crossing', 'collision_severity', 'is_rush_hour', 'is_weekend', 'is_dark', 'is_bad_weather']

#The full feature set used for model training
Model_features = class_features + ['number_of_casualties']

#Columns that were stored as objects
Numeric_object_cols = ['junction_control', 'second_road_class', 'second_road_number', 'pedestrian_crossing_human_control_historic', 'pedestrian_crossing_physical_facilities_historic', 'pedestrian_crossing', 'road_surface_conditions', 'special_conditions_at_site', 'carriageway_hazards_historic', 'carriageway_hazards', 'urban_or_rural_area', 'did_police_officer_attend_scene_of_accident', 'trunk_road_flag', 'enhanced_severity_collision', 'collision_injury_based', 'first_road_number']

#Just for logging info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)