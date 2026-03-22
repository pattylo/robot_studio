#!/usr/bin/env python3

import time
import numpy as np
from math import pi
import serial
import rclpy
from rclpy.node import Node
from mavros_msgs.msg import RCIn


servo_ids_gan = [0,3]


class Motor:
    def __init__(self, servo_ids=[2, 3], port="/dev/ttyUSB0", window_size=10):
        self.port = port
        try:
            self.ser = serial.Serial(port, baudrate=115200, timeout=0.5)
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"Error opening serial port {port}: {e}")
            raise

        self.servo_ids = servo_ids
        self.num_servos = len(servo_ids)

        print(f"Initialized {self.num_servos} servos: {servo_ids}")

        try:
            for sid in servo_ids:
                self._enable_torque(sid)

            for sid in servo_ids:
                self._motor_mode(sid, 0)
        except Exception as e:
            print(f"Error initializing servos: {e}")
            self.ser.close()
            raise

        self.window_size = window_size
        self.timestamps = np.zeros(window_size)
        self.unwrapped_angles = np.zeros((window_size, self.num_servos))
        self.raw_angles = np.zeros(self.num_servos)
        self.prev_angles = np.zeros(self.num_servos)
        self.motor_vel = np.zeros(self.num_servos)
        self.target_speeds = np.zeros(self.num_servos, dtype=int)
        self.sample_idx = 0
        self.is_initialized = False

    def _make_packet(self, servo_id, cmd, params=[]):
        length = 3 + len(params)
        packet = [0x55, 0x55, servo_id, length, cmd] + params
        checksum = (~sum(packet[2:]) & 0xFF)
        packet.append(checksum)
        return bytes(packet)

    def _enable_torque(self, servo_id):
        packet = self._make_packet(servo_id, 31, [0x01])
        self.ser.write(packet)
        self.ser.flush()
        time.sleep(0.02)

    def _motor_mode(self, servo_id, speed):
        speed = max(-1000, min(1000, speed))

        if speed < 0:
            speed_unsigned = 65536 + speed
        else:
            speed_unsigned = speed

        low_byte = speed_unsigned & 0xFF
        high_byte = (speed_unsigned >> 8) & 0xFF

        packet = self._make_packet(servo_id, 29, [0x01, 0x00, low_byte, high_byte])
        self.ser.write(packet)
        self.ser.flush()
        time.sleep(0.02)

    def _read_position(self, servo_id):
        packet = self._make_packet(servo_id, 28, [])
        self.ser.write(packet)
        self.ser.flush()

        time.sleep(0.01)

        data = self.ser.read(8)
        if len(data) < 8:
            return None

        if data[0] != 0x55 or data[1] != 0x55:
            return None

        pos_low = data[5]
        pos_high = data[6]
        position = pos_low | (pos_high << 8)

        return position / 100.0 * np.pi / 180.0

    def read_angles(self):
        for i, sid in enumerate(self.servo_ids):
            ang = self._read_position(sid)
            if ang is not None:
                self.raw_angles[i] = ang

    def update_velocity(self):
        current_time = time.time()
        self.timestamps[self.sample_idx] = current_time

        for i in range(self.num_servos):
            angle_diff = self.raw_angles[i] - self.prev_angles[i]
            if angle_diff > pi:
                angle_diff -= 2 * pi
            elif angle_diff < -pi:
                angle_diff += 2 * pi

            if self.is_initialized:
                self.unwrapped_angles[self.sample_idx, i] = (
                    self.unwrapped_angles[(self.sample_idx - 1) % self.window_size, i] + angle_diff
                )
            else:
                self.unwrapped_angles[self.sample_idx, i] = self.raw_angles[i]

            self.prev_angles[i] = self.raw_angles[i]

        if self.is_initialized:
            oldest_idx = (self.sample_idx + 1) % self.window_size
            dt = self.timestamps[self.sample_idx] - self.timestamps[oldest_idx]
            if dt > 0:
                for i in range(self.num_servos):
                    dangle = (
                        self.unwrapped_angles[self.sample_idx, i]
                        - self.unwrapped_angles[oldest_idx, i]
                    )
                    self.motor_vel[i] = dangle / dt

        self.sample_idx = (self.sample_idx + 1) % self.window_size
        if self.sample_idx == 0:
            self.is_initialized = True

    def set_motor_speed(self, speeds):
        for i, (servo_id, speed) in enumerate(zip(self.servo_ids, speeds)):
            speed_clamped = int(np.clip(speed, -1000, 1000))
            self.target_speeds[i] = speed_clamped
            self._motor_mode(servo_id, speed_clamped)

    def update(self):
        self.read_angles()
        self.update_velocity()

    def stop(self):
        for servo_id in self.servo_ids:
            self._motor_mode(servo_id, 0)

    def __del__(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()


class ServoControlNode(Node):
    def __init__(self):
        super().__init__('servo_control_node')

        self.motor = Motor(servo_ids=servo_ids_gan, port="/dev/ttyUSB0")

        self.desired_manual = np.zeros(2)

        self.rc_sub = self.create_subscription(
            RCIn,
            '/mavros/rc/in',
            self.rc_callback,
            10
        )

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.lala_i = 0
        self.lala_rate = 0

        self.get_logger().info('ctrl via RC')

    def differential_ik(self, vel):
        v_fwd_cmd = vel[0]
        v_turn_cmd = vel[1]

        w_left = (v_fwd_cmd - v_turn_cmd)
        w_right = (v_fwd_cmd + v_turn_cmd)
        return np.array([-w_right, w_left])

    def rc_callback(self, msg):
        if msg.channels[6] < 2000:
            self.desired_manual = np.zeros(2)
        else:
            k = 100.0

            vel = np.array([
                -k * (msg.channels[1] - 1515.0) / (2015.0 - 1015.0) * 2,
                k * (msg.channels[0] - 1515.0) / (2015.0 - 1015.0) * 2
            ])

            self.desired_manual = self.differential_ik(vel)

    def control_loop(self):
        # limits (deg)
        min_deg_0, max_deg_0 = 3.84, 6.57
        min_deg_3, max_deg_3 = 9.28, 11.94

        angles_deg = self.motor.raw_angles / np.pi * 180.0

        if not hasattr(self, "dir_0"):
            self.dir_0 = 1

        # motor 0 drives base direction
        if angles_deg[0] >= max_deg_0:
            self.dir_0 = -1
        elif angles_deg[0] <= min_deg_0:
            self.dir_0 = 1

        # motor 3 tries to be opposite
        dir_3 = -self.dir_0

        # enforce bounds for motor 3 (override if needed)
        if angles_deg[1] >= max_deg_3:
            dir_3 = -1
        elif angles_deg[1] <= min_deg_3:
            dir_3 = 1

        speed_0 = 200 * self.dir_0
        speed_3 = 200 * dir_3

        self.motor.set_motor_speed([speed_0, speed_3])
        self.motor.update()

        print("angles (deg):", angles_deg)

    def shutdown(self):
        self.motor.stop()
        self.motor.ser.close()


def main():
    rclpy.init()

    node = ServoControlNode()

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