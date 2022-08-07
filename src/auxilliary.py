import json
import numpy as np


class ndarrayEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "_type": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "data": obj.flatten().tolist(),
            }
        return super(ndarrayEncoder, self).default(obj)


class ndarrayDecoder(json.JSONDecoder):
    """JSON decoder for numpy arrays"""

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Hook for object in decoder

        :arg obj: Object
        """
        if "_type" not in obj:
            return obj
        datatype = obj["_type"]
        if datatype == "ndarray":
            return np.reshape(np.array(obj["data"], dtype=obj["dtype"]), obj["shape"])
        return obj
