import cv2
import numpy as np
from ultralytics import YOLO
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
        # self.rmat = np.array([[0, -1, 0], 
        #                     [0, 0, -1], 
        #                     [1, 0, 0]], dtype=np.float32)
        # self.tvec = np.array([[0], [0], [0]], dtype=np.float32)
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
        self.camera_coordinate = np.array([coordinate.x, coordinate.y, coordinate.z], dtype=np.float32)
        self.image_coordinate = np.array([0, 0], dtype=np.float32)
        self.width = 250
        self.height = 425
        self.red_balls = 0
        self.blue_balls = 0
        self.purple_balls = 0
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
        self.silos = {"a": Silo(Coordinate(0, 0, 100, 0)), "b": Silo(Coordinate(0, 0, 100, 0)), "c": Silo(Coordinate(0, 0, 100, 0)), "d": Silo(Coordinate(0, 0, 100, 0)), "e": Silo(Coordinate(0, 0, 100, 0))}

        # 過去の状態を保存するキュー
        self.past_states = queue.Queue()

        # on_timer関数を0.1秒ごとに実行
        self.timer = self.create_timer(0.1, self.on_timer)
        
    def listen_silos_coordinate(self, silos):
        to_frame_rel = 'camera_link'
        for id in ['a', 'b', 'c', 'd', 'e']:
            frame = 'silo_' + id
            try:
                t = self.tf_buffer.lookup_transform(
                    to_frame_rel,
                    frame,
                    rclpy.time.Time())
                silos[id] = Silo(Coordinate(t.transform.translation.x, t.transform.translation.y, 100, 0))
            except TransformException as e:
                return silos
        return silos
        
    
    def on_timer(self):
        # サイロの座標を取得
        self.silos = self.listen_silos_coordinate(self.silos)
        
        # カメラのキャプチャを開始
        cap = cv2.VideoCapture(0)
        # カメラからフレームを取得
        ret, frame = cap.read()
        # カメラのキャプチャを終了(メモリの解放)
        cap.release()
        
        # 現在のサイロ内部の状態を取得
        results = self.get_silos_status(frame)
        self.silos = self.detect_balls(results, self.silos)
        # self.show_silos_in_image(frame)

        # 状態を保存
        if self.past_statuses.qsize() > 10:
            self.past_statuses.get()
            self.past_statuses.put([self.silos["a"], self.silos["b"], self.silos["c"], self.silos["d"], self.silos["d"], self.silos["e"]])
        elif self.past_statuses.qsize() <= 10:
            self.past_statuses.put([self.silos["a"], self.silos["b"], self.silos["c"], self.silos["d"], self.silos["d"], self.silos["e"]]) 
        
        # 過去の状態で最も多い状態を取得し、publish
        past_statuses = self.past_statuses.queue
        public_silos = [None] * NUMBER_OF_SILOS
        for i in range(NUMBER_OF_SILOS):
            past_status = []
            for j in range(len(past_statuses)):
                past_status.append(past_statuses[j][i]) 
            counter = Counter(map(tuple, past_status))
            most_common_status = max(counter, key=counter.get)
            public_silos[i] = list(most_common_status)

        # 状態を詰めてpublish
        msg = SilosStatus()
        msg.a.red = public_silos["a"].red_balls
        msg.a.blue = public_silos["a"].blue_balls
        msg.a.purple = public_silos["a"].purple_balls
        msg.b.red = public_silos["b"].red_balls
        msg.b.blue = public_silos["b"].blue_balls
        msg.b.purple = public_silos["b"].purple_balls
        msg.c.red = public_silos["c"].red_balls
        msg.c.blue = public_silos["c"].blue_balls
        msg.c.purple = public_silos["c"].purple_balls
        msg.d.red = public_silos["d"].red_balls
        msg.d.blue = public_silos["d"].blue_balls
        msg.d.purple = public_silos["d"].purple_balls
        msg.e.red = public_silos["e"].red_balls
        msg.e.blue = public_silos["e"].blue_balls
        msg.e.purple = public_silos["e"].purple_balls

        self.publisher.publish(msg)

        return

    # サイロの状態を観測
    def get_silos_status(self, frame):
        # 画像上のサイロの座標を計算
        self.silos = self.convert_camera_to_image_coordinate(self.silos, self.camera)
        # 画像内にサイロがあるか確認
        self.silos = self.check_detection(self.silos)
        
        #画像内のサイロをYOLOで検出
        model = YOLO("../best.pt")
        results = model(frame, show=True)
    
        return results
    
    # 予測結果からボールの色と数を検出
    def detect_balls(self, results, silos):
        silos_x_center = []
        silos_balls = []
        for result in results:
            # サイロの中心座標を取得
            for xyxy in result.boxes.xyxy:
                silos_x_center.append((xyxy[0] + xyxy[2]) / 2)
            # 検出したサイロのクラスを見て、内部のボールを検出
            for cls in result.boxes.cls:
                if cls == 0:
                    silos_balls.append(["none", "none", "none"])
                # ボールが1つの場合
                elif cls == 1:
                    silos_balls.append(["red", "none", "none"])
                elif cls == 2:
                    silos_balls.append(["blue", "none", "none"])
                elif cls == 3:
                    silos_balls.append(["purple", "none", "none"])
                # ボールが2つの場合
                elif cls == 4:
                    silos_balls.append(["red", "red", "none"])
                elif cls == 5:
                    silos_balls.append(["blue", "blue", "none"])
                elif cls == 6:
                    silos_balls.append(["purple", "purple", "none"])
                elif cls == 7:
                    silos_balls.append(["red", "blue", "none"])
                elif cls == 8:
                    silos_balls.append(["red", "purple", "none"])
                elif cls == 9:
                    silos_balls.append(["blue", "purple", "none"])
                # ボールが3つの場合
                elif cls == 10:
                    silos_balls.append(["red", "red", "red"])
                elif cls == 11:
                    silos_balls.append(["blue", "blue", "blue"])
                elif cls == 12:
                    silos_balls.append(["purple", "purple", "purple"])
                elif cls == 13:
                    silos_balls.append(["red", "red", "blue"])
                elif cls == 14:
                    silos_balls.append(["red", "red", "purple"])
                elif cls == 15:
                    silos_balls.append(["blue", "blue", "red"])
                elif cls == 16:
                    silos_balls.append(["blue", "blue", "purple"])
                elif cls == 17:
                    silos_balls.append(["purple", "purple", "red"])
                elif cls == 18:
                    silos_balls.append(["purple", "purple", "blue"])
                elif cls == 19:
                    silos_balls.append(["red", "blue", "purple"])
                    
        # YOLOで検出したサイロの中心座標と自己位置をもとにしたサイロの中心座標から、YOLOで検出したサイロがどのサイロに対応するかを判定
        # その後、サイロにボールの数を記録
        for silo_x_center, silo_balls in zip(silos_x_center, silos_balls):
            for silo in silos:
                if abs(silo_x_center - silo.image_coordinate[0]) < 20 and silo.detection_flag == True:
                    silo.red_balls = silo_balls.count("red")
                    silo.blue_balls = silo_balls.count("blue")
                    silo.purple_balls = silo_balls.count("purple")
                    break
                    
        return silos
    
    def show_silos_in_image(self, frame):
        # サイロ位置を示したカメラ画像を表示
        for silo in self.silos:
            if silo.detection_flag == True:
                cv2.circle(frame, (int(silo.bottom_image_coordinate[0][0]), int(silo.bottom_image_coordinate[1][0])), 10, (0, 0, 255), thickness=cv2.FILLED)
                cv2.circle(frame, (int(silo.top_image_coordinate[0][0]), int(silo.top_image_coordinate[1][0])), 10, (0, 0, 255), thickness=cv2.FILLED)
        cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected', frame)

        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()
            rclpy.shutdown()

    #歪みを考慮した座標変換
    def convert_camera_to_image_coordinate(silos, camera):
        for silo in silos:
            x = silo.camera_coordinate[0] / silo.camera_coordinate[2]
            y = silo.camera_coordinate[1] / silo.camera_coordinate[2]
            r_squared_2 = x ** 2 + y ** 2
            normalized_camera_coordinate = np.array([x * (1 + camera.k1 * r_squared_2 + camera.k2 * r_squared_2 ** 2 + camera.k3 * r_squared_2 ** 3) + 2 * camera.p1 * x * y + camera.p2 * (r_squared_2 + 2 * x ** 2),
                                                    y * (1 + camera.k1 * r_squared_2 + camera.k2 * r_squared_2 ** 2 + camera.k3 * r_squared_2 ** 3) + 2 * camera.p2 * x * y + camera.p1 * (r_squared_2 + 2 * y ** 2),
                                                    1],
                                                    dtype=np.float32)
            
            silo.image_coordinate = np.matmul(camera.camera_matrix, normalized_camera_coordinate)
            
        return silos

    def check_detection(silos):
        for silo in silos:
            if(0 < silo.image_coordinate[0] < 640 and 0 < silo.image_coordinate[1] < 480):
                silo.detection_flag = True
            else:
                silo.detection_flag = False
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