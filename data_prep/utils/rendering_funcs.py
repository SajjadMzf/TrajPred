import numpy as np
import cv2
import os

def initialize_representation(
        width:'Image width', 
        height:'Image height',
        rep_dtype:'Representation dtype (bool or int)',
        filled_value:'Filled value in representation(e.g. True if dtype is bool)',
        occlusion:'Whether to model occlusion or not')->'list of three channels of representation(vehicles, lane marking, observability status)':
        veh_channel = np.zeros(shape=(height, width, 1), dtype=rep_dtype)
        lane_channel = np.zeros(shape=(height, width, 1), dtype=rep_dtype)
        obs_channel = np.zeros(shape=(height, width, 1), dtype=rep_dtype)
        if occlusion == False:
            obs_channel[:,:,:] = filled_value
        return veh_channel, lane_channel, obs_channel

def draw_lane_markings(lane_channel, width, lower_lines, upper_lines, lines_width, filled_value, dash_lines):
        
        # Continues lines
        lane_channel[int(upper_lines[0]):(int(upper_lines[0]) + lines_width), 0:width] = filled_value
        lane_channel[int(upper_lines[-1]):(int(upper_lines[-1]) + lines_width), 0:width] = filled_value
        lane_channel[int(lower_lines[0]):(int(lower_lines[0]) + lines_width), 0:width] = filled_value
        lane_channel[int(lower_lines[-1]):(int(lower_lines[-1]) + lines_width), 0:width] = filled_value
        
        # Broken Lines
        filled = int(dash_lines[0])
        total_line = int((dash_lines[0]+ dash_lines[1]))
        for line in upper_lines[1:-1]:
            for i in range(int(width/total_line)):
                lane_channel[int(line):(int(line) + lines_width), (i*total_line):(i*total_line+filled)] = filled_value

        for line in lower_lines[1:-1]:
            for i in range(int(width/total_line)):
                lane_channel[int(line):(int(line) + lines_width), (i*total_line):(i*total_line+filled)] = filled_value
        
        return lane_channel
      

def draw_vehicle(image, x, y, width, height, filled_value):
    image[y:(y+height), x:(x+width)] = filled_value
    return image

def crop_image(
    image:'uncropped image', 
    image_width:'uncropped image width', 
    image_height:'uncropped image height', 
    tv_x:'crop center x position', 
    tv_y:'crop center y position', 
    tv_lane_markings:'lane marking of the TV side of the road', 
    driving_dir:'TV driving dir',
    tv_lane_ind,
    cropped_height,
    cropped_width,
    lines_width,
    filled_value
    ):
    cropped_img = np.zeros(shape=(cropped_height, cropped_width, 3), dtype=np.uint8)
    left_border = int(np.clip(tv_x-cropped_width/2, 0, image_width))
    right_border = int(np.clip(cropped_width/2 + tv_x, 0, image_width))
    
    
    up_lane_ind = tv_lane_ind-1 if tv_lane_ind-1>0 else 0
    down_lane_ind = tv_lane_ind+2 if tv_lane_ind+2<len(tv_lane_markings) else -1
    up_border = int(tv_lane_markings[up_lane_ind] - lines_width)
    down_border = int(tv_lane_markings[down_lane_ind] + lines_width)

    cropped_left_border = int(left_border-tv_x +cropped_width/2)
    cropped_right_border = int(right_border-tv_x+cropped_width/2)
    cropped_up_border = int(up_border-tv_y+cropped_height/2)
    cropped_down_border = int(down_border-tv_y+cropped_height/2)
    if cropped_up_border<0 or cropped_left_border<0 or cropped_down_border>cropped_height or cropped_right_border>cropped_width:
        valid = False
        return cropped_img, valid

    cropped_img[cropped_up_border:cropped_down_border,cropped_left_border:cropped_right_border,:] = image[up_border:down_border, left_border:right_border, :]
    
    valid = True if np.all(cropped_img[int(cropped_height/2), int(cropped_width/2),2] == filled_value) else False
    
    
    if driving_dir == 2:
        cropped_img = np.flip(cropped_img, 0)
        cropped_img = np.flip(cropped_img, 1)
    
    return cropped_img, valid
    
    
def save_image_sequence(
    v_id:'ID of the TV',
    img_frames:'Actual frame numbers of each image in the sequence',
    images:'Image sequence',
    save_dir:'Save directory',
    file_num
    ):
    folder_name = os.path.join(save_dir, 'file_' + str(file_num).zfill(2) + '_v_' + str(v_id) + '_fr_' + str(img_frames[0]).zfill(6))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for fr, image in enumerate(images):
        
        file_dir = os.path.join(folder_name, str(img_frames[fr]).zfill(6)+'.png')
        if not cv2.imwrite(file_dir, image.astype(np.int8)*255):
            raise Exception("Could not write image: " + file_dir)   

    
    
