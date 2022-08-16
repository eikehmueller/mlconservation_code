"""Auxilliary code for encoding/decoding numpy arrays for json IO

This is used to save the model weights of the Lagrangian NN models in nn_models.py since
JSON can not automatically serialise numpy arrays.
"""
import json
import numpy as np


class ndarrayEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays"""

    def default(self, o):
        """Construct new instance"""
        if isinstance(o, np.ndarray):
            return {
                "_type": "ndarray",
                "shape": list(o.shape),
                "dtype": str(o.dtype),
                "data": o.flatten().tolist(),
            }
        return super().default(o)


class ndarrayDecoder(json.JSONDecoder):
    """JSON decoder for numpy arrays"""

    def __init__(self, *args, **kwargs):
        """Construct new instance"""
        json.JSONDecoder.__init__(self, object_hook=self.numpy_hook, *args, **kwargs)

    def numpy_hook(self, obj):
        """Hook for object in decoder

        :arg obj: Object
        """
        if "_type" not in obj:
            return obj
        datatype = obj["_type"]
        if datatype == "ndarray":
            return np.reshape(np.array(obj["data"], dtype=obj["dtype"]), obj["shape"])
        return obj
