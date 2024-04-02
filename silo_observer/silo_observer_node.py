import cv2
import numpy as np
import queue
from collections import Counter

import rclpy
from rclpy.node import Node
from silo_msgs.msg import SilosStatus

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from geometry_msgs.msg import TransformStamped

NUMBER_OF_SILOS = 5
SILO_CAPACITY = 3

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Coordinate:
    def __init__(self, x, y, z, theta):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta

class Camera:
    def __init__(self):
        # 外部パラメータ
        self.rmat = np.array([[0, -1, 0], 
                            [0, 0, -1], 
                            [1, 0, 0]], dtype=np.float32)
        self.tvec = np.array([[0], [0], [0]], dtype=np.float32)
    # 内部パラメータ
    # Logiccolのc922
    # focal_length = 0.001
    # k1 = 0.03520446031433724
    # k2 = -0.2621147575929849
    # p1 = 0.004920860634892838
    # p2 = 0.007969216065437846
    # k3 = -0.1871326332054414
    # ELECOMカメラ
    focal_length = 0.00415
    k1 = 0.0970794248992087
    k2 = 0.46852992832469376
    p1 = -0.0027402493655248367
    p2 = -0.002055751211595421
    k3 = -12.963018944283235
    # 内部パラメータ行列
    # Logiccolのc922
    # camera_matrix = cv2.Mat(np.array([[1422.092372366652, 0.0, 994.0655146868652], [0.0, 1422.6878709473806, 521.7945002394441], [0.0, 0.0, 1.0]], dtype=np.float32))
    # ELECOMカメラ
    camera_matrix = np.array([[970.7808694146526, 0.0, 385.0122379475739], [0.0, 970.1929411781452, 230.67852825871415], [0.0, 0.0, 1.0]], dtype=np.float32)

class Silo:
    def __init__(self, coordinate):
        self.top_coordinate = np.array([[coordinate.x], [coordinate.y], [525]], dtype=np.float32)
        self.bottom_coordinate = np.array([[coordinate.x], [coordinate.y], [100]], dtype=np.float32)
        self.top_camera_coordinate = None
        self.bottom_camera_coordinate = None
        self.top_image_coordinate = None
        self.bottom_image_coordinate = None

        self.width = 250
        self.height = 425
        self.balls = ["None", "None", "None"]
        self.detection_flag = False

class SiloObserver(Node):
    def __init__(self):
        super().__init__('silo_observer_node')
        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter(
          'target_frame', 'camera_link').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(SilosStatus, 'silos_status', 1)

        self.camera = Camera()
        self.silos = [Silo(Coordinate(6000, 11500, 100, 0)), Silo(Coordinate(6000, 10750, 100, 0)), Silo(Coordinate(6000, 10000, 100, 0)), Silo(Coordinate(6000, 9250, 100, 0)), Silo(Coordinate(6000, 8500, 100, 0))]

        # 過去の状態を保存するキュー
        self.past_states = queue.Queue()

        # on_timer関数を0.1秒ごとに実行
        self.timer = self.create_timer(0.1, self.on_timer)
    
    def on_timer(self):
        # transformの取得に使用する変数にフレーム名を格納する。
        # from_frame_rel = self.target_frame
        to_frame_rel = 'camera_link'

        try:
            # t = self.tf_buffer.lookup_transform(
            #     to_frame_rel,
            #     from_frame_rel,
            #     rclpy.time.Time())
            # # カメラの外部パラメータ(ここは機体座標と、機体の姿勢から逐次的に求める必要がある)
            # quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            # # 機体座標系をx軸周りに90度回転させ、カメラ座標系へ
            # self.camera.rmat = np.matmul(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32), quaternion_to_rotation_matrix(quat))
            # # カメラ座標系での機体位置の座標を求め、機体中心からカメラまでのベクトルを足すことでカメラ座標系でのカメラの位置を表す
            # self.camera.tvec = np.matmul(self.camera.rmat, np.array([[t.transform.translation.x], [t.transform.translation.y], [0]], dtype=np.float32)) + np.array([[0], [0], [50]], dtype=np.float32)

            # self.get_logger().info(f'camera rmat : {self.camera.rmat} : camera.tvec: {self.camera.tvec}')

            # カメラ座標系でのサイロの座標を取得
            count = 0
            for i in ['a', 'b', 'c', 'd', 'e']:
                frame = 'silo_' + i
                t = self.tf_buffer.lookup_transform(
                    to_frame_rel,
                    frame,
                    rclpy.time.Time())
                self.silos[count] = Silo(Coordinate(t.transform.translation.x, t.transform.translation.y, t.transform.translation.z, 0))
                # self.get_logger().info(f'{self.silos[count].bottom_coordinate}')
                count += 1
            
        except TransformException as ex:
            # self.get_logger().info(
            #     f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            # return

        # 現在の状態を取得
        self.observe_silos()

        # 過去の状態を保存
        if self.past_states.qsize() > 10:
            self.past_states.get()
            self.past_states.put([self.silos[0].balls, self.silos[1].balls, self.silos[2].balls, self.silos[3].balls, self.silos[4].balls])
        elif self.past_states.qsize() <= 10:
            self.past_states.put([self.silos[0].balls, self.silos[1].balls, self.silos[2].balls, self.silos[3].balls, self.silos[4].balls]) 
        
        # 過去の状態で最も多い状態を取得
        past_states = self.past_states.queue
        for i in range(NUMBER_OF_SILOS):
            past_state = []
            for j in range(len(past_states)):
                past_state.append(past_states[j][i]) 
            counter = Counter(map(tuple, past_state))
            most_common_state = max(counter, key=counter.get)
            self.silos[i].balls = list(most_common_state)

        # 状態を詰めてpublish
        msg = SilosStatus()
        msg.a.red = self.silos[0].balls.count("red")
        msg.a.blue = self.silos[0].balls.count("blue")
        msg.a.purple = self.silos[0].balls.count("purple")
        msg.b.red = self.silos[1].balls.count("red")
        msg.b.blue = self.silos[1].balls.count("blue")
        msg.b.purple = self.silos[1].balls.count("purple")
        msg.c.red = self.silos[2].balls.count("red")
        msg.c.blue = self.silos[2].balls.count("blue")
        msg.c.purple = self.silos[2].balls.count("purple")
        msg.d.red = self.silos[3].balls.count("red")
        msg.d.blue = self.silos[3].balls.count("blue")
        msg.d.purple = self.silos[3].balls.count("purple")
        msg.e.red = self.silos[4].balls.count("red")
        msg.e.blue = self.silos[4].balls.count("blue")
        msg.e.purple = self.silos[4].balls.count("purple")

        self.publisher.publish(msg)

        return

    # サイロの状態を観測
    def observe_silos(self):
        # カメラのキャプチャを開始
        cap = cv2.VideoCapture(0)
        # カメラからフレームを取得
        ret, frame = cap.read()

        # カメラ画像上のサイロの頂点の座標を取得
        # self.silos = world_to_camera_coordinate(self.silos, self.camera)
        self.silos = camera_to_image_coordinate(self.silos, self.camera)

        # 画像を切り取ってボール検出
        self.silos, cut_imgs = cut_image(self.silos, frame)
        self.silos = detect_balls(self.silos, cut_imgs)

        # # カットした画像表示
        # xx = 0
        # for img in cut_imgs:
        #     cv2.namedWindow('cut'+str(xx), cv2.WINDOW_NORMAL)
        #     cv2.imshow('cut'+str(xx), img)
        #     xx += 1

        # サイロ位置を示したカメラ画像を表示
        for silo in self.silos:
            if silo.detection_flag == True:
                cv2.circle(frame, (int(silo.bottom_image_coordinate[0][0]), int(silo.bottom_image_coordinate[1][0])), 10, (0, 0, 255), thickness=cv2.FILLED)
                cv2.circle(frame, (int(silo.top_image_coordinate[0][0]), int(silo.top_image_coordinate[1][0])), 10, (0, 0, 255), thickness=cv2.FILLED)
        cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected', frame)

        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()
            cap.release()
            rclpy.shutdown()
        #メモリを解放して終了するためのコマンド
        cap.release()
        
# # クォータニオンから回転行列に変換
# def quaternion_to_rotation_matrix(quat):
#     x = quat[0]
#     y = quat[1]
#     z = quat[2]
#     w = quat[3]
#     rmat = np.array([[2*w**2 + 2*x**2 - 1, 2*x*y - 2*z*w, 2*x*z + 2*y*w], [2*x*y + 2*z*w, 2*w**2 + 2*y**2 - 1, 2*y*z - 2*x*w], [2*x*z - 2*y*w, 2*y*z + 2*x*w, 2*w**2 + 2*z**2 - 1]], dtype=np.float32)
    
#     return rmat

# カメラ画像上のサイロの頂点の座標を取得
def world_to_camera_coordinate(silos, camera):
    for silo in silos:
        silo.bottom_camera_coordinate = np.matmul(camera.rmat, silo.bottom_coordinate) + camera.tvec
        silo.top_camera_coordinate = np.matmul(camera.rmat, silo.top_coordinate) + camera.tvec

    return silos

#歪みを考慮した座標変換
def camera_to_image_coordinate(silos, camera):
    for silo in silos:
        bottom_x = silo.bottom_camera_coordinate[0] / silo.bottom_camera_coordinate[2]
        bottom_y = silo.bottom_camera_coordinate[1] / silo.bottom_camera_coordinate[2]
        r_squared_2 = bottom_x ** 2 + bottom_y ** 2
        silo.bottom_camera_coordinate[0] = bottom_x * (1 + camera.k1 * r_squared_2 + camera.k2 * r_squared_2 ** 2 + camera.k3 * r_squared_2 ** 3) + 2 * camera.p1 * bottom_x * bottom_y + camera.p2 * (r_squared_2 + 2 * bottom_x ** 2)
        silo.bottom_camera_coordinate[1] = bottom_y * (1 + camera.k1 * r_squared_2 + camera.k2 * r_squared_2 ** 2 + camera.k3 * r_squared_2 ** 3) + 2 * camera.p2 * bottom_x * bottom_y + camera.p1 * (r_squared_2 + 2 * bottom_y ** 2)
        silo.bottom_camera_coordinate[2] = 1
        
        top_x = silo.top_camera_coordinate[0] / silo.top_camera_coordinate[2]
        top_y = silo.top_camera_coordinate[1] / silo.top_camera_coordinate[2]
        r_squared_2 = top_x ** 2 + top_y ** 2
        silo.top_camera_coordinate[0] = top_x * (1 + camera.k1 * r_squared_2 + camera.k2 * r_squared_2 ** 2 + camera.k3 * r_squared_2 ** 3) + 2 * camera.p1 * top_x * top_y + camera.p2 * (r_squared_2 + 2 * top_x ** 2)
        silo.top_camera_coordinate[1] = top_y * (1 + camera.k1 * r_squared_2 + camera.k2 * r_squared_2 ** 2 + camera.k3 * r_squared_2 ** 3) + 2 * camera.p2 * top_x * top_y + camera.p1 * (r_squared_2 + 2 * top_y ** 2)
        silo.top_camera_coordinate[2] = 1

        silo.bottom_image_coordinate = np.matmul(camera.camera_matrix, silo.bottom_camera_coordinate)
        silo.top_image_coordinate = np.matmul(camera.camera_matrix, silo.top_camera_coordinate)

    return silos

# カメラ画像からサイロ部分のみを切り取る
def cut_image(silos, image):
    cut_imgs = []
    for silo in silos:
        if(0 < silo.top_image_coordinate[0] < 640 and 0 < silo.top_image_coordinate[1] < 480 and 0 < silo.bottom_image_coordinate[0] < 640 and 0 < silo.bottom_image_coordinate[1] < 480):
            silo.detection_flag = True
            cut_imgs.append(image[int(silo.top_image_coordinate[1]) : int(silo.bottom_image_coordinate[1]), int(silo.bottom_image_coordinate[0] - 1) : int(silo.bottom_image_coordinate[0] + 1)])
        else:
            silo.detection_flag = False
    
    return silos, cut_imgs

# サイロ部分の画像からボールを検出
def detect_balls(silos, cut_imgs):
    # HSV範囲（調整が必要な場合は変更）
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    blue_lower = np.array([100, 70, 50])
    blue_upper = np.array([140, 255, 255])
    purple_lower = np.array([240, 30, 50])
    purple_upper = np.array([255, 50, 255])

    detected_count = 0
    for silo in silos:
        if silo.detection_flag == True:
            # 画像を縦方向に3分割
            height, width = cut_imgs[detected_count].shape[:2]
            for i in range(SILO_CAPACITY):
                # 各領域の範囲を計算
                start_y = i * int(height / 3)
                end_y = (i + 1) * int(height / 3)

                if start_y <= 0 or end_y <= 0:
                    break

                # 領域を切り取り
                region = cut_imgs[detected_count][start_y:end_y, :]

                # BGR形式からHSV形式に変換
                region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

                # それぞれの色の領域を抽出
                red_mask = cv2.inRange(region_hsv, red_lower, red_upper)
                blue_mask = cv2.inRange(region_hsv, blue_lower, blue_upper)
                purple_mask = cv2.inRange(region_hsv, purple_lower, purple_upper)

                # それぞれの色のピクセル数をカウント
                red_pixels = cv2.countNonZero(red_mask)
                blue_pixels = cv2.countNonZero(blue_mask)
                purple_pixels = cv2.countNonZero(purple_mask)

                # 画像内のピクセル数
                total_pixels = region_hsv.shape[0] * region_hsv.shape[1]

                # それぞれの色の割合を計算
                red_ratio = red_pixels / total_pixels
                blue_ratio = blue_pixels / total_pixels
                purple_ratio = purple_pixels / total_pixels
                
                if red_ratio > 0.4:
                    silo.balls[i] = "red"
                elif blue_ratio > 0.4:
                    silo.balls[i] = "blue"
                elif purple_ratio > 0.4:
                    silo.balls[i] = "purple"
                else:
                    silo.balls[i] = "None"
            detected_count += 1
                
    return silos

def main(args=None):
    rclpy.init(args=args)

    silo_observer = SiloObserver()

    try:
        rclpy.spin(silo_observer)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()