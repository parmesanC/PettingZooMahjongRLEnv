"""
自定义 JSON 编码器，支持 numpy 数组序列化
"""
import json
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """支持 numpy 数组的 JSON 编码器"""

    def default(self, obj):
        """处理 numpy 类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # 其他类型交给父类处理（会抛出 TypeError）
        return super().default(obj)
