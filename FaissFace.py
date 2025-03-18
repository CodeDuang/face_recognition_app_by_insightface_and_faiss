import os

import cv2
import faiss
import numpy as np

# from . import faiss_util
# from . import img_to_vector
import faiss_util
import img_to_vector
# from utils.config import faiss_db_path as db_path


class FaissFace:
    def __init__(
        self,
        faiss_db_path: str = None,
        insightface_file: str = None,
        dimension: int = 512,
    ):
        if not faiss_db_path:
            faiss_db_path = db_path

        if not os.path.isfile(faiss_db_path):  # 新建数据库
            faiss_util.make_faiss_db(faiss_db_path, dimension)

        self.faiss_db_path = faiss_db_path
        self.faiss_index = faiss.read_index(faiss_db_path)  # 数据库对象
        self.insigtface_app = img_to_vector.get_insightface(
            models_path=insightface_file, model_name="buffalo_s"
        )  # 模型对象

    def insert_face(self, img: cv2.Mat):
        """
        插入人脸
        :param img: 人脸图片
        :return: 人脸id
        """
        # 获取所有用户id，找到未被使用的id
        id = 0
        if self.faiss_index.ntotal > 0:  # 判断数据库是否为空
            id_lst = faiss_util.get_all_id(self.faiss_index)
            id_lst = sorted(id_lst)
            for i in range(len(id_lst)):
                if id_lst[i] > i:
                    id = i
                    break
            else:
                id = len(id_lst)

        # 图片->人脸向量
        faces_vector_list, _ = img_to_vector.img_to_vector(img, self.insigtface_app)
        if len(faces_vector_list) != 1:
            return -1

        # 人脸向量->faiss数据库
        faiss_util.add_vector_to_faiss_db(self.faiss_index, faces_vector_list, [id])

        # 向量数据库存档
        faiss_util.save_faiss_db(self.faiss_index, self.faiss_db_path)
        # print(f"\n已存入人脸，当前faiss数据库中存储向量个数为{self.faiss_index.ntotal}\n")
        return id

    def reco_face(self, img: cv2.Mat):
        """
        识别人脸
        :param img: 人脸图片
        :return: { #返回不是这样的
            "faiss_id": [人脸id1,人脸id2,人脸id3...]
            "bbox": [[x,y,w,h],[x,y,w,h],[x,y,w,h]...]
            "distance": [距离1,距离2,距离3]
            "accurate": [准确度1,准确度2,准确度3]
        }
        """
        # 图片->face向量，face方框
        faces_vector_list, faces_bbox_list = img_to_vector.img_to_vector(
            img, self.insigtface_app
        )
        
        if not faces_vector_list:
            return []
        
        if self.faiss_index.ntotal <= 0: #faiss是空的，但是图上有人脸待识别
            result = [
            {
                "x": round(x[0], 3),
                "y": round(x[1], 3),
                "w": round(x[2] - x[0], 3),
                "h": round(x[3] - x[1], 3),
                "faiss_id": -1,
                "distance": float('inf'),
                "accurate": 0,
            }
            for x in faces_bbox_list
            ]
            return result

        # face向量->faiss数据库搜索
        ids_lst, dis_lst, acc_lst = faiss_util.search_faiss_db(
            self.faiss_index, faces_vector_list, threshold=1.24
        )

        result = [
            {
                "x": round(x[0], 3),
                "y": round(x[1], 3),
                "w": round(x[2] - x[0], 3),
                "h": round(x[3] - x[1], 3),
                "faiss_id": y,
                "distance": round(z, 3),
                "accurate": round(k, 3),
            }
            for x, y, z, k in zip(faces_bbox_list, ids_lst, dis_lst, acc_lst)
        ]


        return result

    def delete_face(self, faiss_id: int):
        """
        删除人脸
        :param faiss_id: 人脸id
        :return:
        """
        befor_ntotal = self.faiss_index.ntotal  # 删前数量
        # 根据id删除向量
        faiss_util.remove_vector_from_faiss_db(self.faiss_index, [faiss_id])

        # 向量数据库存档
        faiss_util.save_faiss_db(self.faiss_index, self.faiss_db_path)

        now_ntotal = self.faiss_index.ntotal  # 删后数量
        if now_ntotal < befor_ntotal:
            return {"code": 0, "message": "删除成功"}
        else:
            return {"code": -1, "message": f"删除失败，不存在id为{faiss_id}的向量"}
        
    def draw(self, image, results):
        for result in results:
            user_id = result["faiss_id"]
            cv2.rectangle(image, (int(result["x"]), int(result["y"])), (int(result["w"]+result["x"]), int(result["h"]+result["y"])), (0, 255, 0), 2)
            # cv2.putText(image, str(user_id), (result["x"], result["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return image



if __name__ == "__main__":
    f = FaissFace(insightface_file="./",faiss_db_path="./faiss_db/faiss.db")

    # 插入人脸
    id = f.insert_face(cv2.imread("./img/a.jpg"))
    

    # 打开笔记本摄像头
    cap = cv2.VideoCapture(0)  # 参数0表示默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    
    # # 设置帧间隔（例如每10帧取一张画面）
    # frame_interval = 10
    # frame_count = 0

    # while True:
    #     print(123)
    #     ret, frame = cap.read()  # 读取一帧画面
    #     if not ret:
    #         print("无法读取画面")
    #         break

    #     # 每隔指定帧数取出一张画面
    #     if frame_count % frame_interval == 0:
    #         print(f"处理第 {frame_count} 帧画面")
    #         # 在这里对画面进行分析
    #         # 例如：显示画面
    #         # cv2.imshow("Frame", frame)
    #         # 识别人脸
    #         re = f.reco_face(frame)
    #         print(re)
    #     frame_count += 1

    #     # 按下 'q' 键退出循环
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # 释放摄像头资源并关闭窗口
    # cap.release()
    # cv2.destroyAllWindows()

    #识别
    re = f.reco_face(cv2.imread("./img/b.jpg"))
    print(re)

    #保存识别结果
    img = f.draw(cv2.imread("./img/b.jpg"),re)
    cv2.imwrite('resout.jpg', img)

    # 删除人脸
    f.delete_face(id)
