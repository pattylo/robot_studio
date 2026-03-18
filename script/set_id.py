#!/usr/bin/env python3
import time
import serial

def set_servo_id(port, current_id, new_id):
    ser = serial.Serial(port, baudrate=115200, timeout=0.1)
    time.sleep(0.1)

    print(f"Changing servo ID {current_id} → {new_id}")

    # Command 13 = ID write
    length = 4
    packet = [0x55, 0x55, current_id, length, 13, new_id]

    checksum = (~sum(packet[2:]) & 0xFF)
    packet.append(checksum)

    ser.write(bytes(packet))
    time.sleep(0.1)

    ser.close()
    print("Done.")

if __name__ == "__main__":
    set_servo_id("/dev/ttyUSB0", current_id=1, new_id=3)