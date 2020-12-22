import lmdb
import sys
import numpy as np

env = lmdb.open(sys.argv[1])

with env.begin() as txn:
    with txn.cursor() as cursor:
        for key, value in cursor:
            print(key.decode())
            print(np.frombuffer(value, dtype=np.float32).shape)