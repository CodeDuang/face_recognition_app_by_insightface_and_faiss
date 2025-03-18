import faiss
import numpy as np


def make_faiss_db(path: str, dimension: int = 512) -> None:
    """
    创建一个faiss向量数据库,并使用IndexIDMap来管理ID映射
    参数：
    - path: 数据库路径名称
    - dimension: 向量维度
    返回: None
    """
    index = faiss.IndexFlatL2(
        dimension
    )  # 将我们数据的维度信息传递给 faiss.IndexFlatL2 函数，建立一个空的索引容器
    # 创建一个IndexIDMap来管理ID映射
    index_with_id = faiss.IndexIDMap(index)
    faiss.write_index(index_with_id, path)  # 将索引容器保存到文件中
    return index_with_id


def save_faiss_db(index: faiss.Index, path: str) -> None:
    """
    保存faiss向量数据库
    参数：
    - index: faiss索引容器对象
    - path: 数据库路径名称
    返回: None
    """

    faiss.write_index(index, path)  # 将索引容器保存到文件中


def load_faiss_db(path: str) -> faiss.Index:
    """
    加载本地已有的faiss向量数据库
    参数：
    - path: 数据库路径名称
    返回:
    - index: faiss索引容器对象
    """

    index = faiss.read_index(path)  # 加载之前保存的索引
    return index


def add_vector_to_faiss_db(
    index: faiss.Index, vectors: np.ndarray, ids: np.ndarray
) -> None:
    """
    向faiss数据库中添加向量
    参数：
    - index: faiss索引容器对象
    - vectors: 向量列表，每个向量是一个numpy数组
    - ids: 向量对应的ID列表，每个ID是一个整数
    返回: None
    """

    if not isinstance(vectors, np.ndarray):  # 判断vectors的类型是否符合要求
        vectors = np.array(vectors, dtype=np.float32)  # 转化为numpy数组
    if not isinstance(ids, np.ndarray):  # 判断ids的类型是否符合要求
        ids = np.array(ids, dtype=np.int64)  # 转化为numpy数组

    if vectors.shape[1] != index.d:  # 判断vectors的维度是否符合要求
        raise ValueError(
            f"注意，您的向量维度{vectors.shape[1]}不符合数据库要求{index.d}，无法存入请检查！"
        )
    index.add_with_ids(vectors, ids)  # 向索引中添加向量


def remove_vector_from_faiss_db(index: faiss.Index, id_lst) -> None:
    """
    从faiss数据库中删除向量
    参数：
    - index: faiss索引容器对象
    - id_lst: 需删除的向量下标列表，每个下标是一个整数
    返回: None
    """

    idx_to_rm = np.array(id_lst)  # 转化为numpy数组

    index.remove_ids(idx_to_rm)  # 从索引中删除对应下标的向量


def search_faiss_db(
    index: faiss.Index, query_vector: np.ndarray, k: int = 1, threshold: float = 1.24
):
    """
    (注意：务必保证向量已归一化，否则阈值不起效；
           暂时只支持搜索单一向量，即query_vector.shape=(1,d))
    在faiss数据库中搜索向量
    参数：
    - index: faiss索引容器对象
    - query_vector: 查询向量，是一个numpy数组
    - k: 查询返回的向量个数，默认为1
    - threshold: 阈值，默认为1.24
    返回:
    - outt_id: 查询到的向量下标列表，每个下标是一个整数(可能为空，表示数据库中没有匹配的人脸信息)
    - outt_dis: 查询到的向量距离列表，每个距离是一个浮点数
    - outt_accurate: 查询到的向量准确度列表，每个准确度是一个浮点数
    """

    # 类型检查
    if not isinstance(query_vector, np.ndarray):
        query_vector = np.array(query_vector, dtype=np.float32)
    # 返回最近的k个向量
    distances, labels = index.search(query_vector, k)
    # print(f"distances:{distances},labels:{labels}")

    outt_id = []  # 返回的用户id列表
    outt_dis = []  # 返回距离列表
    for n in range(query_vector.shape[0]):  # 遍历一张图的多个脑袋
        for i in range(k - 1, -1, -1):
            if distances[n][i] < threshold:  # 小于阈值表示距离近
                # print(f"和阈值{threshold}对比后：返回下标:{labels[n].tolist()[:i+1]}")
                outt_id.append(labels[n].tolist()[: i + 1])
                outt_dis.append(distances[n].tolist()[: i + 1])
                break
        else:
            outt_dis.append(distances[n].tolist()[: i + 1])
            outt_id.append([-1])
    outt_id = [i[0] for i in outt_id]
    outt_dis = [i[0] for i in outt_dis]

    # 准确率
    outt_accurate = [  (3.125-i**2)/3.125 if i<=threshold  else 0.97/(x**3)  for i in outt_dis] #准确度计算公式
    return outt_id, outt_dis, outt_accurate


def get_all_id(
    index: faiss.Index,
):  # TODO：该函数认为，两个向量距离不会超过4*d+5.0，以此作为阈值，则可检索所有向量id,请验证
    """
    返回faiss数据库中所有向量的id(注意所有向量应归一化，否则该函数不起效，阈值也不起效)
    参数：
    - index: faiss索引容器对象
    返回:
    - id_lst: 所有向量id列表，每个id是一个整数
    """
    d = index.d
    lon = index.ntotal
    threshold = 4 * d + 5.0 
    
    distances, labels = index.search(np.array([[1 for _ in range(d)]], dtype=np.float32), lon)
    # outt, _,_ = search_faiss_db(
    #     index=index,
    #     query_vector=np.array([[1 for _ in range(d)]], dtype=np.float32),
    #     k=lon,
    #     threshold=threshold,
    # )
    return labels[0]


if __name__ == "__main__":
    """
    使用举例
    """

    dimension = 2  # 向量维度
    path = "./src/faiss/database/faiss_db"  # faiss数据库路径
    index = make_faiss_db(path=path, dimension=dimension)  # 创建一个faiss向量数据库
    print(
        f"新创建的向量数据库，维度为{dimension}，存储向量个数为{index.ntotal}，路径为{path}\n"
    )

    add_vector_to_faiss_db(
        index,
        vectors=[
            [2.68586492e-03, 2.84447912e-02],
            [-4.02664095e-02, 8.37431010e-03],
            [-4.02664095e-02, 8.37431010e-03],
            [2.68586492e-03, 2.84447912e-02],
            [-4.02664095e-02, 8.37431010e-03],
        ],
        ids=[100, 200, 300, 400, 500],
    )  # 向faiss数据库中添加10个随机生成的向量
    print(f"添加5个向量后，存储向量个数为{index.ntotal}\n")

    save_faiss_db(index, path)  # 保存faiss数据库
    print(f"保存faiss数据库到{path}\n")

    remove_vector_from_faiss_db(index, [100, 500])  # 从faiss数据库中删除向量
    print(f"删除100,500号向量后，存储向量个数为{index.ntotal}\n")

    index = load_faiss_db(path)  # 加载faiss数据库
    print(f"本地加载向量数据库后，存储向量个数为{index.ntotal}\n")

    id_lst = get_all_id(index)
    print(f"faiss数据库中所有向量的id为{id_lst}\n")

    x, _, _ = search_faiss_db(
        index,
        np.array(
            [[-4.02664095e-02, 8.37431010e-03], [2.68586492e-03, 2.84447912e-02]],
            dtype=np.float32,
        ),
    )  # 查询向量[1,2]在faiss数据库中的匹配向量
    print(f"查询向量在faiss数据库中的匹配向量为{x}\n")
