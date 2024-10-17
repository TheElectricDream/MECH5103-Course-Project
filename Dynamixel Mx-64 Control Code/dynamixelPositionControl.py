import dynamixelFunctions as dxlf
import time
import math
import csv
from datetime import datetime
import threading

#---------------------------------------------------------------------------------------------#
# DEFINE CONSTANTS

# Set the control table addresses for the MX-64
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112

# Data Byte Length
LEN_GOAL_POSITION = 4
LEN_PRESENT_POSITION = 4

# Protocol version
PROTOCOL_VERSION = 2.0  # Protocol version 2.0

# Default setting
DXL1_ID = 1  # Dynamixel ID: 1
DXL2_ID = 2  # Dynamixel ID: 2
BAUDRATE = 1000000  # Dynamixel baudrate: 1Mbps
DEVICENAME = 'COM3'  # Port name (Windows: COM3)

TORQUE_ENABLE = 1  # Value for enabling the torque
TORQUE_DISABLE = 0  # Value for disabling the torque
DXL_MINIMUM_POSITION = 1534  # ~135 deg - Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION = 2558  # ~225 deg - Dynamixel will rotate between this value

# Constants
UNITS_PER_DEGREE = 4096 / 360  # Units per degree for the Dynamixel
DEGREE_PER_UNIT = 360 / 4096  # Degree per unit

# Generate positions in degrees and units
positions_in_degrees = list(range(150, 200, 2))  # Degrees from 150 to 180 inclusive
positions_in_degrees_flipped = list(range(198, 148, -2))  # Flipped degrees for reverse scan
positions_in_degrees_horz = list(range(148, 220, 3))  # Degrees for horizontal scan
positions_in_units = [int(degree * UNITS_PER_DEGREE) for degree in positions_in_degrees]
positions_in_units_flipped = [int(degree * UNITS_PER_DEGREE) for degree in positions_in_degrees_flipped]
positions_in_units_horz = [int(degree * UNITS_PER_DEGREE) for degree in positions_in_degrees_horz]

#---------------------------------------------------------------------------------------------#
# CHOOSE PAUSE OPTION

pause_option = input("Choose the pause mode:\n1 - Press Enter to proceed to next point\n2 - Pause for 1 second between steps\nEnter 1 or 2: ")

#---------------------------------------------------------------------------------------------#
# INITIALIZE DYNAMIXEL

# Initialize PortHandler and PacketHandler instances
port_num, packet_handler, group_sync_write = dxlf.initialize_dynamixels(
    DEVICENAME,
    PROTOCOL_VERSION,
    BAUDRATE,
    ADDR_GOAL_POSITION,
    LEN_GOAL_POSITION
)

# Enable dynamixel torque
dxlf.enable_torque(DXL1_ID, packet_handler, port_num, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
dxlf.enable_torque(DXL2_ID, packet_handler, port_num, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# Set the profile velocity for both Dynamixels
desired_velocity = 30  # Adjust according to your needs
dxlf.set_profile_velocity(DXL1_ID, desired_velocity, packet_handler, port_num, ADDR_PROFILE_VELOCITY)
dxlf.set_profile_velocity(DXL2_ID, desired_velocity, packet_handler, port_num, ADDR_PROFILE_VELOCITY)

#---------------------------------------------------------------------------------------------#
# DATA LOGGING THREAD

# Get the current datetime when the script starts
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a filename with the current datetime appended
filename = f'encoder_data_{start_time}.csv'

logging_active = True  # Flag to control logging

def log_data():
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Encoder Angle DXL1 (degrees)", "Encoder Angle DXL2 (degrees)"])
        
        while logging_active:
            # Read present positions
            angle_dxl1_degrees = dxlf.read_present_position(DXL1_ID, packet_handler, port_num, ADDR_PRESENT_POSITION)
            angle_dxl2_degrees = dxlf.read_present_position(DXL2_ID, packet_handler, port_num, ADDR_PRESENT_POSITION)

            # Write the data (timestamp and encoder angles) to the CSV
            current_time = time.time()
            writer.writerow([current_time, angle_dxl1_degrees, angle_dxl2_degrees])
            
            time.sleep(1)  # Log data every 100ms (adjust as needed)

# Start the logging thread
logging_thread = threading.Thread(target=log_data)
logging_thread.start()

#---------------------------------------------------------------------------------------------#
# FUNCTION TO MOVE DYNAMIXEL WITH RETRY MECHANISM

def move_to_position(dxl_id, position, packet_handler, group_sync_write):
    while True:
        try:
            group_sync_write.clearParam()  # Clear any previous parameters
            dxlf.set_goal_position(dxl_id, group_sync_write, position)
            dxlf.sync_write_goal_position(packet_handler, group_sync_write)
            group_sync_write.clearParam()
            return  # Success
        except Exception as e:
            print(f"Error moving Dynamixel {dxl_id} to position {position}: {e}")
            print("Retrying...")
            group_sync_write.clearParam()  # Clear any previous parameters
            dxlf.set_goal_position(dxl_id, group_sync_write, position)
            dxlf.sync_write_goal_position(packet_handler, group_sync_write)
            group_sync_write.clearParam()
            return  # Success

#---------------------------------------------------------------------------------------------#
# START CONTROL LOOP

flip = False
PointsCounterHorz = 0 
PointsCounterVert = 0

try:
    for position_dxl2 in positions_in_units_horz:
        # Move Dynamixel 2 to the new position
        move_to_position(DXL2_ID, position_dxl2, packet_handler, group_sync_write)

        # Pause according to selected option
        if pause_option == '1':
            input("Press Enter to proceed to the next DXL2 position...")
            PointsCounterVert+=1
        elif pause_option == '2':
            time.sleep(1)

        # Perform raster scan with DXL1
        if not flip:
            # Scan forward
            for position_dxl1 in positions_in_units:
                move_to_position(DXL1_ID, position_dxl1, packet_handler, group_sync_write)

                if pause_option == '1':
                    input("Press Enter to proceed to the next point...")
                    PointsCounterHorz+=1
                elif pause_option == '2':
                    time.sleep(1)

            flip = True  # Set to flip after scan forward

        else:
            # Scan backward (flipped)
            for position_dxl1 in positions_in_units_flipped:
                move_to_position(DXL1_ID, position_dxl1, packet_handler, group_sync_write)

                if pause_option == '1':
                    input("Press Enter to proceed to the next point...")
                    PointsCounterHorz+=1
                elif pause_option == '2':
                    time.sleep(1)

            flip = False  # Reset flip after scan backward

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    logging_active = False  # Stop logging
    logging_thread.join()  # Wait for the logging thread to finish

    # Disable torque and close port
    dxlf.enable_torque(DXL1_ID, packet_handler, port_num, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    dxlf.enable_torque(DXL2_ID, packet_handler, port_num, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

    #Print the points
    print(PointsCounterHorz)
    print(PointsCounterVert)
