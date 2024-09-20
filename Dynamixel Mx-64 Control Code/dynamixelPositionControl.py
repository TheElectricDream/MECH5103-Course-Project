#---------------------------------------------------------------------------------------------#
# IMPORT MODULES

import dynamixelFunctions as dxlf
import time
import math
import csv
from datetime import datetime

#---------------------------------------------------------------------------------------------#
# DEFINE CONSTANTS

# Set the control table addresses for the MX-64
ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132
ADDR_PRESENT_POSITION   = 132

# Data Byte Length
LEN_GOAL_POSITION       = 4
LEN_PRESENT_POSITION    = 4

# Protocol version
PROTOCOL_VERSION        = 2.0                 # Protocol version 2.0

# Default setting
DXL1_ID                 = 1                   # Dynamixel ID: 1
DXL2_ID                 = 2                   # Dynamixel ID: 2
BAUDRATE                = 1000000             # Dynamixel baudrate: 1Mbps
DEVICENAME              = 'COM3'              # Port name (Windows: COM3)

TORQUE_ENABLE           = 1                   # Value for enabling the torque
TORQUE_DISABLE          = 0                   # Value for disabling the torque
DXL_MINIMUM_POSITION    = 0                   # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION    = 4095                # Dynamixel will rotate between this value
DXL_MOVING_STATUS_THRESHOLD = 20              # Dynamixel moving status threshold

# Constants
AMPLITUDE   = (3072 - 1024) / 2  # Amplitude of the sine wave
MIDPOINT    = (3072 + 1024) / 2   # Midpoint between 1024 and 3072
FREQUENCY   = 0.5                # Frequency of the sine wave (adjust as needed)
PERIOD      = 1.0 / FREQUENCY       # Period of the sine wave

#---------------------------------------------------------------------------------------------#
# INITIALIZE DYNAMIXEL

# Initialize PortHandler and PacketHandler instances
port_num, packet_handler, group_sync_write = dxlf.initialize_dynamixels(DEVICENAME, 
                                                                        PROTOCOL_VERSION, 
                                                                        BAUDRATE, 
                                                                        ADDR_GOAL_POSITION, 
                                                                        LEN_GOAL_POSITION)

# Enable dynamixel torque
dxlf.enable_torque(DXL1_ID, packet_handler, port_num, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
dxlf.enable_torque(DXL2_ID, packet_handler, port_num, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

#---------------------------------------------------------------------------------------------#
# START CONTROL LOOP

# Get the current datetime when the script starts
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a filename with the current datetime appended
filename = f'encoder_data_{start_time}.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(["Timestamp", "Encoder Angle DXL1 (degrees)", "Encoder Angle DXL2 (degrees)"])

    while True:    

        # Get the current time for the sine wave calculation
        current_time = time.time()

        # Calculate the sine wave for this time step (sin value oscillates between -1 and 1)
        sine_value = math.sin(2 * math.pi * FREQUENCY * current_time)

        # Map the sine value to the range 1024 to 3072
        goal_position = int(MIDPOINT + AMPLITUDE * sine_value)

        # Set goal positions for both Dynamixels without overwriting group_sync_write
        dxlf.set_goal_position(DXL1_ID, group_sync_write, goal_position)
        dxlf.set_goal_position(DXL2_ID, group_sync_write, goal_position)

        # Write both positions
        dxlf.sync_write_goal_position(packet_handler, group_sync_write)

        # Read present position
        angle_dxl1 = dxlf.read_present_position(DXL1_ID, packet_handler, port_num, ADDR_PRESENT_POSITION)
        angle_dxl2 = dxlf.read_present_position(DXL2_ID, packet_handler, port_num, ADDR_PRESENT_POSITION)

        # Write the data (timestamp and encoder angles) to the CSV
        writer.writerow([current_time, angle_dxl1, angle_dxl2])

        # Clear parameters for the next sync write
        group_sync_write.clearParam()

        # Delay to match the update frequency
        time.sleep(0.01)  # Adjust the sleep time as needed for smooth operation

