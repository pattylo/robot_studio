#!/usr/bin/env python3

import time
import numpy as np
from math import pi
import serial
import rclpy
from rclpy.node import Node
from mavros_msgs.msg import RCIn
import time


servo_ids_gan=[0,1,2,3]

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
        
        # enable torque for all servos first
        try:
            for sid in servo_ids:
                self._enable_torque(sid)
            
            # stop all servos initially
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
        """Create a command packet."""
        length = 3 + len(params)
        packet = [0x55, 0x55, servo_id, length, cmd] + params
        checksum = (~sum(packet[2:]) & 0xFF)
        packet.append(checksum)
        return bytes(packet)
    
    def _enable_torque(self, servo_id):
        
        # command 31: enable torque
        packet = self._make_packet(servo_id, 31, [0x01])
        self.ser.write(packet)
        self.ser.flush()
        time.sleep(0.02)
    
    def _motor_mode(self, servo_id, speed):
        speed = max(-1000, min(1000, speed))
        
        # convert to unsigned 16-bit for negative speeds
        if speed < 0:
            speed_unsigned = 65536 + speed  # Two's complement
        else:
            speed_unsigned = speed
        
        low_byte = speed_unsigned & 0xFF
        high_byte = (speed_unsigned >> 8) & 0xFF
        
        # command 29: motor mode, format: [enable=1, reserved=0, speed_low, speed_high]
        packet = self._make_packet(servo_id, 29, [0x01, 0x00, low_byte, high_byte])
        self.ser.write(packet)
        self.ser.flush()
        time.sleep(0.02)
    
    def read_angles(self):
        """Read current angles from all servos (stub - requires position mode)."""
        # Note: Reading angles requires servos to be in position mode, not motor mode
        # For now, just keep previous values
        pass
    
    def update_velocity(self):
        current_time = time.time()
        self.timestamps[self.sample_idx] = current_time
        
        for i in range(self.num_servos):
            # unwrap angle
            angle_diff = self.raw_angles[i] - self.prev_angles[i]
            if angle_diff > pi:
                angle_diff -= 2 * pi
            elif angle_diff < -pi:
                angle_diff += 2 * pi
            
            if self.is_initialized:
                self.unwrapped_angles[self.sample_idx, i] = self.unwrapped_angles[(self.sample_idx - 1) % self.window_size, i] + angle_diff
            else:
                self.unwrapped_angles[self.sample_idx, i] = self.raw_angles[i]
            
            self.prev_angles[i] = self.raw_angles[i]
        
        if self.is_initialized:
            # calculate velocity from window
            oldest_idx = (self.sample_idx + 1) % self.window_size
            dt = self.timestamps[self.sample_idx] - self.timestamps[oldest_idx]
            if dt > 0:
                for i in range(self.num_servos):
                    dangle = self.unwrapped_angles[self.sample_idx, i] - self.unwrapped_angles[oldest_idx, i]
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
        
        # subject to connection
        self.motor = Motor(servo_ids=servo_ids_gan, port="/dev/ttyUSB0")
        
        self.desired_manual = np.zeros(2)
        
        # sub to input (rc)
        self.rc_sub = self.create_subscription(
            RCIn,
            '/mavros/rc/in',
            self.rc_callback,
            10
        )
        
        # timer for ctrl
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.lala_i   = 0
        self.lala_rate = 400
        
        self.get_logger().info('ctrl via RC')
    
    def differential_ik(self, vel):
        v_fwd_cmd = vel[0]
        v_turn_cmd = vel[1]
        
        w_left = (v_fwd_cmd - v_turn_cmd)
        w_right = (v_fwd_cmd + v_turn_cmd)
        return np.array([-w_right, w_left])
    
    def rc_callback(self, msg):
        
        if msg.channels[6] < 2000: 
            # switch here
            self.desired_manual = np.zeros(2)
        else:
            k = 100.0
            
            vel = np.array([
                -k * (msg.channels[1] - 1515.0) / (2015.0 - 1015.0) * 2,
                k * (msg.channels[0] - 1515.0) / (2015.0 - 1015.0) * 2
            ])
            
            self.desired_manual = self.differential_ik(vel)
            
    
    def control_loop(self):
        
        # motor_speeds = np.clip(self.desired_manual * 10, -1000, 1000).astype(int)
        motor_speeds = np.array([self.lala_rate, self.lala_rate,self.lala_rate,self.lala_rate]) * 0.5
        self.lala_i = self.lala_i + 1
        
        if self.lala_i % 10 == 0:
            self.lala_rate = self.lala_rate * (-1)
            
        # motor_speeds = np.array([-200]) 
        
        # self.lala_rate = self.lala_rate * 0.5
        
        print(motor_speeds)
        self.motor.set_motor_speed(motor_speeds)
        self.motor.update()
    
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