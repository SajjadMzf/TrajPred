import pandas
import numpy as np
import os
import pickle

# exid params
Y2LANE = 'y2lane'
LANE_WIDTH = 'laneWidth'
# TRACK FILE
BBOX = "bbox"
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "height"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
FRONT_SIGHT_DISTANCE = "frontSightDistance"
BACK_SIGHT_DISTANCE = "backSightDistance"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
PRECEDING_X_VELOCITY = "precedingXVelocity"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"

SV_IDs = [
        PRECEDING_ID, 
        FOLLOWING_ID,
        LEFT_PRECEDING_ID, 
        LEFT_ALONGSIDE_ID,
        LEFT_FOLLOWING_ID, 
        RIGHT_PRECEDING_ID, 
        RIGHT_ALONGSIDE_ID, 
        RIGHT_FOLLOWING_ID
        ]
# STATIC FILE
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
NUM_FRAMES = "numFrames"
CLASS = "class"
DRIVING_DIRECTION = "drivingDirection"
TRAVELED_DISTANCE = "traveledDistance"
MIN_X_VELOCITY = "minXVelocity"
MAX_X_VELOCITY = "maxXVelocity"
MEAN_X_VELOCITY = "meanXVelocity"
MIN_DHW = "minDHW"
MIN_THW = "minTHW"
MIN_TTC = "minTTC"
NUMBER_LANE_CHANGES = "numLaneChanges"

# VIDEO META
ID = "id"
FRAME_RATE = "frameRate"
LOCATION_ID = "locationId"
SPEED_LIMIT = "speedLimit"
MONTH = "month"
WEEKDAY = "weekDay"
START_TIME = "startTime"
DURATION = "duration"
TOTAL_DRIVEN_DISTANCE = "totalDrivenDistance"
TOTAL_DRIVEN_TIME = "totalDrivenTime"
N_VEHICLES = "numVehicles"
N_CARS = "numCars"
N_TRUCKS = "numTrucks"
UPPER_LANE_MARKINGS = "upperLaneMarkings"
LOWER_LANE_MARKINGS = "lowerLaneMarkings"


def read_track_csv(input_path, pickle_path, reload = True, group_by = 'frames', fr_div = 1):
    """
    This method reads the tracks file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the tracks csv file.
    :return: a list containing all tracks as dictionaries.
    """
    if reload == False and os.path.exists(pickle_path):
            pickle_in = open(pickle_path, "rb")
            #print('Record read from pickle file:', pickle_path)
            frames = pickle.load(pickle_in)
            
            return frames 
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(input_path)
    selected_frames = (df.frame%fr_div == 0).real.tolist()
    df = df.loc[selected_frames]#.to_frame()
    #frame_group = df.groupby([FRAME], sort = False)

    # Use groupby to aggregate track info. Less error prone than iterating over the data.
    if group_by == 'frames':
        grouped = df.groupby([FRAME], sort=True)    
    elif group_by == 'tracks':
        grouped = df.groupby([TRACK_ID], sort=True)
    else:
        raise('Unknown group_by parameter')
    current_group = 0
    groups = [None] * grouped.ngroups
    for group_id, rows in grouped:
        bounding_boxes = np.transpose(np.array([rows[X].values,
                                                rows[Y].values,
                                                rows[WIDTH].values,
                                                rows[HEIGHT].values]))
        groups[current_group] = {TRACK_ID: rows[TRACK_ID].values, 
                                 FRAME: rows[FRAME].values,
                                 BBOX: bounding_boxes,
                                 X: rows[X].values,
                                 Y: rows[Y].values,
                                 Y2LANE: rows[Y2LANE].values,
                                 LANE_WIDTH: rows[LANE_WIDTH].values,
                                 X_VELOCITY: rows[X_VELOCITY].values,
                                 Y_VELOCITY: rows[Y_VELOCITY].values,
                                 X_ACCELERATION: rows[X_ACCELERATION].values,
                                 Y_ACCELERATION: rows[Y_ACCELERATION].values,
                                 HEIGHT: rows[HEIGHT].values,
                                 WIDTH: rows[WIDTH].values,
                                 PRECEDING_ID: rows[PRECEDING_ID].values,
                                 FOLLOWING_ID: rows[FOLLOWING_ID].values,
                                 LEFT_FOLLOWING_ID: rows[LEFT_FOLLOWING_ID].values,
                                 LEFT_ALONGSIDE_ID: rows[LEFT_ALONGSIDE_ID].values,
                                 LEFT_PRECEDING_ID: rows[LEFT_PRECEDING_ID].values,
                                 RIGHT_FOLLOWING_ID: rows[RIGHT_FOLLOWING_ID].values,
                                 RIGHT_ALONGSIDE_ID: rows[RIGHT_ALONGSIDE_ID].values,
                                 RIGHT_PRECEDING_ID: rows[RIGHT_PRECEDING_ID].values,
                                 LANE_ID: rows[LANE_ID].values
                                 }
        current_group = current_group + 1
    pickle_out = open(pickle_path, "wb")
    pickle.dump(groups, pickle_out)
    pickle_out.close()
    return groups


def read_static_info(input_static_path):
    """
    This method reads the static info file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(input_static_path)

    # Declare and initialize the static_dictionary
    static_dictionary = {}

    # Iterate over all rows of the csv because we need to create the bounding boxes for each row
    for i_row in range(df.shape[0]):
        track_id = int(df[TRACK_ID][i_row])
        static_dictionary[track_id] = {TRACK_ID: track_id,
                                       INITIAL_FRAME: int(df[INITIAL_FRAME][i_row]),
                                       FINAL_FRAME: int(df[FINAL_FRAME][i_row]),
                                       NUM_FRAMES: int(df[NUM_FRAMES][i_row]),
                                       DRIVING_DIRECTION: float(df[DRIVING_DIRECTION][i_row])
                                       }
    return static_dictionary


def read_meta_info(input_meta_path):
    """
    This method reads the video meta file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the video meta csv file.
    :return: the meta dictionary containing the general information of the video
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(input_meta_path)

    # Declare and initialize the extracted_meta_dictionary
    extracted_meta_dictionary = {ID: int(df[ID][0]),
                                 FRAME_RATE: int(df[FRAME_RATE][0]),
                                 LOCATION_ID: int(df[LOCATION_ID][0]),
                                 N_VEHICLES: int(df[N_VEHICLES][0]),
                                 UPPER_LANE_MARKINGS: np.fromstring(df[UPPER_LANE_MARKINGS][0], sep=";"),
                                 LOWER_LANE_MARKINGS: np.fromstring(df[LOWER_LANE_MARKINGS][0], sep=";")}
    return extracted_meta_dictionary
