目前这并不算是一个开箱即用的项目(后续还会继续完善)，虽然没有前端页面，但该有的功能都有，需要帮助请提issues\(@^0^@)/
## 致谢
##### 参考项目

- 人脸画框+特征提取使用开源项目insightface：https://github.com/deepinsight/insightface
- 特征向量的存储和比对，使用开源项目faiss: https://github.com/facebookresearch/faiss

##### 贡献者

<a href="https://github.com/codeduang/face_recognition_app_by_insightface_and_faiss/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codeduang/face_recognition_app_by_insightface_and_faiss" />
</a>

## 更新日志


- [x] **2025.3.12**：README.md文档补充了完整目录结构。

- [x] **2025.3.7**：封装了人脸检测模型和faiss向量数据库，提供函数：添加人脸身份、删除人脸身份、获取所有人脸身份、检索图片中所有人脸信息。


## 使用手册

##### python环境
python推荐使用3.x（python3.8没有语法List\[int]）  
faiss-gpu #如果没有cuda，请使用 faiss-cpu
opencv-python  
insightface  
onnxruntime  

#### 目录结构（请通过./models/buffalo_s文件夹下的路径下载所需的insightface模型）
face_recognition_app_by_insightface_and_faiss  
├── FaissFace.py  
├── faiss_util.py  
├── img_to_vector.py  
├── models  
│   └── buffalo_s  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 1k3d68.onnx  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 2d106det.onnx  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── det_500m.onnx  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── genderage.onnx  
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── w600k_mbf.onnx  
└── README.md  

