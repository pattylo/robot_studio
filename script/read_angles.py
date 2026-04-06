#!/usr/bin/env python3

import time
import serial


SERVO_IDS = [0, 1, 2, 3]
PORT = "/dev/ttyUSB0"


def make_packet(servo_id, cmd, params=[]):
    length = 3 + len(params)
    packet = [0x55, 0x55, servo_id, length, cmd] + params
    checksum = (~sum(packet[2:]) & 0xFF)
    packet.append(checksum)
    return bytes(packet)


def read_position(ser, servo_id):
    # cmd 28 = read position
    packet = make_packet(servo_id, 28, [])
    ser.write(packet)
    ser.flush()

    time.sleep(0.02)

    data = ser.read(8)
    if len(data) < 8:
        return None

    if data[0] != 0x55 or data[1] != 0x55:
        return None

    pos_low = data[5]
    pos_high = data[6]
    position = pos_low | (pos_high << 8)

    return position / 100.0  # degrees


def main():
    try:
        ser = serial.Serial(PORT, baudrate=115200, timeout=0.1)
        time.sleep(0.1)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return

    print("Reading servo angles...")

    try:
        while True:
            angles = []
            for sid in SERVO_IDS:
                ang = read_position(ser, sid)
                angles.append(ang)

            print(angles)
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()