#!/usr/bin/env python3

import time
import numpy as np
import serial
import rclpy
from rclpy.node import Node

 
servo_ids_gan = [0,1,2,3]

class Motor:
    def __init__(self, servo_ids=[0], port="/dev/ttyUSB0"):
        self.ser = serial.Serial(port, baudrate=115200, timeout=0.1)
        time.sleep(0.1)

        self.servo_ids = servo_ids
        self.num_servos = len(servo_ids)

        print(f"Initialized servos: {servo_ids}")

        for sid in servo_ids:
            self._enable_torque(sid)
            self._exit_motor_mode(sid)

    def _make_packet(self, servo_id, cmd, params=[]):
        length = 3 + len(params)
        packet = [0x55, 0x55, servo_id, length, cmd] + params
        checksum = (~sum(packet[2:]) & 0xFF)
        packet.append(checksum)
        return bytes(packet)

    def _enable_torque(self, servo_id):
        self.ser.write(self._make_packet(servo_id, 31, [0x01]))
        self.ser.flush()
        time.sleep(0.02)

    def _exit_motor_mode(self, servo_id):
        packet = self._make_packet(servo_id, 29, [0x00, 0x00, 0x00, 0x00])
        self.ser.write(packet)
        self.ser.flush()
        time.sleep(0.02)

    def _enter_motor_mode(self, servo_id, speed=0):
        speed = int(np.clip(speed, -1000, 1000))
        if speed < 0:
            speed = 65536 + speed

        low = speed & 0xFF
        high = (speed >> 8) & 0xFF

        packet = self._make_packet(servo_id, 29, [0x01, 0x00, low, high])
        self.ser.write(packet)
        self.ser.flush()
        time.sleep(0.02)

    def _read_position(self, servo_id):
        self.ser.write(self._make_packet(servo_id, 28, []))
        self.ser.flush()
        time.sleep(0.02)

        data = self.ser.read(8)
        if len(data) < 8:
            return None

        pos = data[5] | (data[6] << 8)
        return pos / 100.0

    def read_all(self):
        angles = []
        for sid in self.servo_ids:
            ang = self._read_position(sid)
            angles.append(ang)
        return np.array(angles)

    def _position_mode(self, servo_id, angle_deg, time_ms=300):
        angle = int(np.clip(angle_deg, 0, 240) * 100)

        pos_low = angle & 0xFF
        pos_high = (angle >> 8) & 0xFF

        time_low = time_ms & 0xFF
        time_high = (time_ms >> 8) & 0xFF

        packet = self._make_packet(
            servo_id,
            1,
            [pos_low, pos_high, time_low, time_high]
        )

        self.ser.write(packet)
        self.ser.flush()

    def set_positions(self, angles):
        for sid, ang in zip(self.servo_ids, angles):
            self._position_mode(sid, ang)

    def enable_velocity_mode(self):
        for sid in self.servo_ids:
            self._enter_motor_mode(sid, 0)

    def close(self):
        self.ser.close()


class ServoNode(Node):
    def __init__(self):
        super().__init__('servo_position_sway')

        self.motor = Motor(servo_ids=servo_ids_gan)

        time.sleep(0.2)
        self.initial_angles = self.motor.read_all()

        print("initial angles (deg):", self.initial_angles)

        # different offsets per motor
        self.offsets = np.array([1.5,1.0,1.0,1.5])   # motor 0: ±5°, motor 3: ±3°
        self.freq = 0.5

        self.timer = self.create_timer(0.05, self.control_loop)

    def control_loop(self):
        t = time.time()

        delta = self.offsets * np.sin(2 * np.pi * self.freq * t)
        # delta[0] = 0.0
        # delta[1] = 0.0
        # delta[2] = 0.0
        # delta[1] = 0.0
        # bias = np.array([0,0,0,1.0])
        target_angles = self.initial_angles + delta  #+ bias
        # + bias

        self.motor.set_positions(target_angles)

    def shutdown(self):
        # Return to initial position before exiting
        self.motor.set_positions(self.initial_angles)
        time.sleep(0.5)  # Wait for servos to reach initial position
        
        self.motor.enable_velocity_mode()
        self.motor.close()


def main():
    rclpy.init()

    node = ServoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()