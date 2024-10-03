import dynamixel_sdk as dxl

# Initialize PortHandler and PacketHandler instances
def initialize_dynamixels(DEVICENAME: str, PROTOCOL_VERSION: float, BAUDRATE: int, ADDR_GOAL_POSITION: int, LEN_GOAL_POSITION: int):
    """
    Initializes the PortHandler and PacketHandler for Dynamixel communication.

    Args:
        DEVICENAME (str): The device name (e.g., COM3 on Windows or /dev/ttyUSB0 on Linux).
        PROTOCOL_VERSION (float): The protocol version used by Dynamixel (e.g., 1.0, 2.0).
        BAUDRATE (int): Baud rate for serial communication (e.g., 1000000 for 1 Mbps).
        ADDR_GOAL_POSITION (int): The address in the control table for setting the goal position.
        LEN_GOAL_POSITION (int): The byte length of the goal position (usually 4 bytes).

    Returns:
        tuple: A tuple containing:
            - port_num (PortHandler): The PortHandler instance for managing the communication port.
            - packet_handler (PacketHandler): The PacketHandler instance for managing data packets.
            - group_sync_write (GroupSyncWrite): The GroupSyncWrite instance for syncing goal position writes.
    """
    # Initialize PortHandler instance
    port_num = dxl.PortHandler(DEVICENAME)

    # Initialize PacketHandler instance
    packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)

    # Open port
    if port_num.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        quit()

    # Set port baudrate
    if port_num.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        quit()

    # Initialize GroupSyncWrite instance
    group_sync_write = dxl.GroupSyncWrite(port_num, packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    return port_num, packet_handler, group_sync_write


# Enable Dynamixel Torque
def enable_torque(dxl_id: int, packet_handler, port_num, ADDR_TORQUE_ENABLE: int, TORQUE_ENABLE: int):
    """
    Enables or disables torque for a given Dynamixel motor.

    Args:
        dxl_id (int): The ID of the Dynamixel motor.
        packet_handler (PacketHandler): The PacketHandler instance used to send/receive packets.
        port_num (PortHandler): The PortHandler instance managing the communication port.
        ADDR_TORQUE_ENABLE (int): The address in the control table for enabling/disabling torque.
        TORQUE_ENABLE (int): The value to enable (1) or disable (0) torque.

    Returns:
        PacketHandler: The updated PacketHandler instance.
    """
    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(port_num, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"{dxl_id}: {packet_handler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"{dxl_id}: {packet_handler.getRxPacketError(dxl_error)}")
    else:
        print(f"Dynamixel {dxl_id} torque enabled")

    return packet_handler


# Set goal position for a specific Dynamixel
def set_goal_position(dxl_id: int, group_sync_write, position: int):
    """
    Sets the goal position of a specific Dynamixel motor.

    Args:
        dxl_id (int): The ID of the Dynamixel motor.
        group_sync_write (GroupSyncWrite): The GroupSyncWrite instance for syncing goal position writes.
        position (int): The goal position to be set (e.g., 0 to 4095 for a full 360-degree rotation).

    Returns:
        None
    """
    position_bytes = [dxl.DXL_LOBYTE(dxl.DXL_LOWORD(position)),
                      dxl.DXL_HIBYTE(dxl.DXL_LOWORD(position)),
                      dxl.DXL_LOBYTE(dxl.DXL_HIWORD(position)),
                      dxl.DXL_HIBYTE(dxl.DXL_HIWORD(position))]

    # Add Dynamixel goal position to the SyncWrite group
    dxl_addparam_result = group_sync_write.addParam(dxl_id, position_bytes)
    if not dxl_addparam_result:
        print(f"[ID:{dxl_id}] groupSyncWrite addparam failed")


# Read current encoder position (angle) from the Dynamixel
def read_present_position(dxl_id: int, packet_handler, port_num, ADDR_PRESENT_POSITION: int) -> float:
    """
    Reads the current encoder position (angle) from the Dynamixel motor.

    Args:
        dxl_id (int): The ID of the Dynamixel motor.
        packet_handler (PacketHandler): The PacketHandler instance used to send/receive packets.
        port_num (PortHandler): The PortHandler instance managing the communication port.
        ADDR_PRESENT_POSITION (int): The address in the control table for reading the present position.

    Returns:
        float: The current position in degrees (0 to 360).
    """
    # Read present position (4 bytes)
    dxl_present_position, dxl_comm_result, dxl_error = packet_handler.read4ByteTxRx(port_num, dxl_id, ADDR_PRESENT_POSITION)
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"Read present position failed for ID:{dxl_id}: {packet_handler.getTxRxResult(dxl_comm_result)}")

    # Convert the raw encoder value to an angle in degrees (0-360)
    position_in_degrees = (dxl_present_position / 4095.0) * 360.0

    return position_in_degrees


# Execute the SyncWrite group to move both Dynamixels
def sync_write_goal_position(packet_handler, group_sync_write):
    """
    Executes the SyncWrite group to move both Dynamixels to their goal positions.

    Args:
        packet_handler (PacketHandler): The PacketHandler instance used to send/receive packets.
        group_sync_write (GroupSyncWrite): The GroupSyncWrite instance containing goal positions to write.

    Returns:
        None
    """
    # Transmit the goal position values
    dxl_comm_result = group_sync_write.txPacket()
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"SyncWrite failed: {packet_handler.getTxRxResult(dxl_comm_result)}")


# Set the profile velocity for smooth motion
def set_profile_velocity(dxl_id: int, profile_velocity: int, packet_handler, port_num, ADDR_PROFILE_VELOCITY: int):
    """
    Sets the profile velocity of a specific Dynamixel motor to control the speed of movement.

    Args:
        dxl_id (int): The ID of the Dynamixel motor.
        profile_velocity (int): The velocity to be set for the motor (0 to 32767, where 0 disables velocity control).
        packet_handler (PacketHandler): The PacketHandler instance used to send/receive packets.
        port_num (PortHandler): The PortHandler instance managing the communication port.
        ADDR_PROFILE_VELOCITY (int): The address in the control table for setting the profile velocity.

    Returns:
        None
    """
    # Write the profile velocity to the motor
    dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(port_num, dxl_id, ADDR_PROFILE_VELOCITY, profile_velocity)
    
    # Check for communication errors
    if dxl_comm_result != dxl.COMM_SUCCESS:
        print(f"{dxl_id}: {packet_handler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"{dxl_id}: {packet_handler.getRxPacketError(dxl_error)}")


