from geometry_msgs.msg import TransformStamped

import rclpy
from rclpy.node import Node

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class StaticSiloCoordinatesPublisher(Node):
    def __init__(self):
        super().__init__('silo_coordinates_broadcaster')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        self.make_transforms()

    def make_transforms(self):
        y = 2000
        for c in ['a', 'b', 'c', 'd', 'e']:
            t = TransformStamped()

            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'silo_' + c

            t.transform.translation.x = float(4000)
            t.transform.translation.y = float(y)
            t.transform.translation.z = float(100)
            t.transform.rotation.x = float(0)
            t.transform.rotation.y = float(0)
            t.transform.rotation.z = float(0)
            t.transform.rotation.w = float(1)

            self.tf_static_broadcaster.sendTransform(t)
            self.tf_static_broadcaster = StaticTransformBroadcaster(self)

            y -= 500


def main():
    rclpy.init()
    node = StaticSiloCoordinatesPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()