import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn import preprocessing

# from insightface.data import get_image


def get_insightface(
    allowed_modules=["detection", "recognition"],
    models_path="/res/xcl/insightface",
    model_name="buffalo_s",
):
    """
    根据参数定义，返回insightface对象
    参数：
    - allowed_modules: 表示使用哪些模型，不同的模型起不同的功能：detection是人脸检测，recognition是识别特征
    - models_path: 模型的路径（该路径下需要有：./models/buffalo_s）
    - model_name: 模型的名字，可选:buffalo_s, buffalo_l
    返回：
    - app: 一个FaceAnalysis类的实例，里面有get和draw_on两个方法，用于检测和绘制检测框
    """
    # 根据参数定义insightface对象，可能有多个模型被加载，根据allowed_modules参数来决定
    app = FaceAnalysis(
        name=model_name,
        allowed_modules=allowed_modules,
        root=models_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))  # 缩放到640x640像素再输入到模型进行处理
    return app


def img_to_vector(img, app):
    """
    输入图片的opencv对象，返回该图片的faces的特征向量和bbox
    参数:
    - img: 图片对象，通过cv2.imread()读取的图片
    - app: insightface对象
    返回：
    - faces_vector_list: 返回一个二维列表，每一个元素都是一个face向量
    - faces_bbox_list: 返回一个二维列表，每一个元素都是一个face的bbox坐标
    """

    # img = ins_get_image(img_path[:-4]) #不用带后缀
    faces = app.get(img)  # 进行检测('bbox'、'det_score'、'embedding')
    faces_vector_list = []
    faces_bbox_list = []
    for face in faces:  # 遍历每一个检测到的人脸
        face_bbox = np.array(face.bbox)  # face.bbox : (4,)
        vector = np.array(face.embedding)  # face.embedding : (512,)

        vector = preprocessing.normalize(vector.reshape(1, -1)).reshape(
            -1
        )  # 归一化，以便和阈值比较 #TODO:如果效果不好，可以不归一化

        faces_vector_list.append(vector)
        faces_bbox_list.append(face_bbox.tolist())
    return faces_vector_list, faces_bbox_list


if __name__ == "__main__":
    """
    使用举例
    """
    # 定义insightface对象
    app = get_insightface(
        allowed_modules=["detection", "recognition"],
        models_path="/res/xcl/iot_project/aiface/src/face",
        model_name="buffalo_s",
    )

    # 读取图片
    img_path = "/res/xcl/iot_project/aiface/images/many_people.jpg"
    img = cv2.imread(img_path)

    # 获取图片的vector，和框框坐标
    img_faces, img_bbox = img_to_vector(img, app)
    # print(img_faces)

    # print(
    #     type(img_faces[0]), img_faces[0].shape
    # )  # 这是图片的第一张人脸的特征向量 # <class 'numpy.ndarray'>
    # print(
    #     type(img_bbox[0]), img_bbox[0].shape
    # )  # 这是图片的第一张人脸的方框坐标 # <class 'numpy.ndarray'>
    n = 1
    img_bbox = np.array(img_bbox,dtype=np.int64)
    for bbox in img_bbox:
        print(bbox)
        cimg = img[bbox[1]-12:bbox[3]+15,bbox[0]-7:bbox[2]+7]
        print(cimg)
        cv2.imwrite("/res/xcl/iot_project/aiface/images/"+str(n)+".jpg",cimg)
        n+=1
