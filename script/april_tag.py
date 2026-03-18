#!/usr/bin/env python3

import cv2
import apriltag

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped, PointStamped

class AprilTagPoseEstimator(Node):
    def __init__(self, camera_matrix, dist_coeffs, tag_size, target_id):
        super().__init__('apriltag_pose_estimator')
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.target_id = target_id
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag25h9"))
        self.tracked_points = []
        self.poses = []
        self.bridge = CvBridge()
        self.pose_pub = self.create_publisher(PoseStamped, '/mavros/mocap/pose', 10)
        self.tag_center_pub = self.create_publisher(PointStamped, '/ugv/tag_center', 10)
        self.image_pub = self.create_publisher(Image, '/processed_image', 10)
        self.image_sub = self.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', self.image_callback, 10)
        
        self.get_logger().info("gan")

    def image_callback(self, msg):
        self.get_logger().info("cam here")
        
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        display_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        self.detect_and_plot(cv_image, display_image)
        
        return                                

    def detect_and_plot(self, cv_image, display_image):
        cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        detections = self.detector.detect(cv_image)
        print(len(detections))
        for detection in detections:
            if detection.tag_id == self.target_id:
                self.publish_pose(detection, display_image)
                self.display_trajectory(display_image)
                break

    def publish_pose(self, detection, display_image):
        img_points = detection.corners
        center_x = np.mean([point[0] for point in img_points])
        center_y = np.mean([point[1] for point in img_points])
        self.tracked_points.append((int(center_x), int(center_y)))

        obj_points = np.array([
            [-self.tag_size / 2, self.tag_size / 2, 0],
            [self.tag_size / 2, self.tag_size / 2, 0],
            [self.tag_size / 2, -self.tag_size / 2, 0],
            [-self.tag_size / 2, -self.tag_size / 2, 0]
        ])
        
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs)
        
        if success:
            # Draw 3D axes on the tag
            self.draw_axes(display_image, rvec, tvec)
            
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            rotation = Rotation.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()  # Returns [x, y, z, w]

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_frame"
            pose_msg.pose.position.x = float(tvec[0])
            pose_msg.pose.position.y = float(tvec[1])
            pose_msg.pose.position.z = float(tvec[2])
            pose_msg.pose.orientation.x = float(quaternion[0])
            pose_msg.pose.orientation.y = float(quaternion[1])
            pose_msg.pose.orientation.z = float(quaternion[2])
            pose_msg.pose.orientation.w = float(quaternion[3])
            self.pose_pub.publish(pose_msg)

            # UGV approach: tag center in image (pixel) for stage-two (pixel) alignment
            center_msg = PointStamped()
            center_msg.header.stamp = pose_msg.header.stamp
            center_msg.header.frame_id = "camera_frame"
            center_msg.point.x = float(center_x)
            center_msg.point.y = float(center_y)
            center_msg.point.z = 0.0
            self.tag_center_pub.publish(center_msg)
    
    def draw_axes(self, image, rvec, tvec):
        """Draw 3D axes on the image"""
        # Define axis points in 3D (origin + axis endpoints)
        axis_length = self.tag_size * 0.5  # Half the tag size for visibility
        axis_points_3d = np.array([
            [0, 0, 0],           # Origin
            [axis_length, 0, 0],  # X-axis (Red)
            [0, axis_length, 0],  # Y-axis (Green)
            [0, 0, axis_length]   # Z-axis (Blue, pointing outward from plane)
        ], dtype=np.float32)
        
        # Project 3D points to 2D image plane
        axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, 
                                              self.camera_matrix, self.dist_coeffs)
        axis_points_2d = axis_points_2d.reshape(-1, 2)
        
        # Convert to integer coordinates
        origin = tuple(axis_points_2d[0].astype(int))
        x_end = tuple(axis_points_2d[1].astype(int))
        y_end = tuple(axis_points_2d[2].astype(int))
        z_end = tuple(axis_points_2d[3].astype(int))
        
        # Draw axes with different colors
        # X-axis: Red
        cv2.line(image, origin, x_end, (0, 0, 255), 3)
        cv2.putText(image, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Y-axis: Green
        cv2.line(image, origin, y_end, (0, 255, 0), 3)
        cv2.putText(image, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Z-axis: Blue
        cv2.line(image, origin, z_end, (255, 0, 0), 3)
        cv2.putText(image, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def display_trajectory(self, image):
        overlay_image = image.copy()
        num_points = len(self.tracked_points)
        
        if num_points < 2:
            return

        for i in range(1, num_points):
            pt1 = self.tracked_points[i - 1]
            pt2 = self.tracked_points[i]
            color = (0, 0, 255)
            cv2.line(image, pt1, pt2, color, 4)

        image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self.image_pub.publish(image_msg)

def main(args=None):
    rclpy.init(args=args)

    camera_matrix = np.array([
        [628.8233032226562, 0, 646.732666015625],
        [0, 628.3672485351562, 364.4508361816406],
        [0, 0, 1]
    ], dtype=np.float32)

    # -0.056777212768793106, 0.06796900182962418, 0.0007022436475381255, 0.0004860123444814235, -0.021817076951265335

    dist_coeffs = np.array([-0.056777212768793106, 0.06796900182962418, 0.0007022436475381255, 0.0004860123444814235], dtype=np.float32)
    tag_size = 0.3
    target_id = 0

    estimator = AprilTagPoseEstimator(camera_matrix, dist_coeffs, tag_size, target_id)
    
    try:
        rclpy.spin(estimator)
    except KeyboardInterrupt:
        pass
    finally:
        estimator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()