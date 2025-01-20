import h5py
import dask.array as da


class H5Reader:
    def __init__(self, path):
        self.path = path
        self.file = h5py.File(path, "r")

    def __getitem__(self, key):
        return H5Var(self.file[key], str(self))
    
    def __del__(self):
        self.file.close()

    def vars(self):
        return list(self.file.keys())
    
    def __repr__(self):
        return f"H5Reader({self.path})"
    

class H5Var:
    def __init__(self, group, reader):
        self.group = group
        self.reader = reader

    def __getitem__(self, key: int):
        return H5Rec(self.group[f"run_{key}"], self.reader, self.group.name, key)
    
    def n(self):
        return len(self.group.keys())
    
    def keys(self):
        return list(self.group.keys())
    
    def __repr__(self):
        return f"{self.reader} : {self.group.name}"

    
class H5Rec:
    def __init__(self, group, reader, name, run):
        self.group = group
        self.reader = reader
        self.name = name
        self.run = run
        datasets = [self.group[ds] for ds in self.group]
        arrays = [da.from_array(ds, chunks=(-1, 1, -1, -1)) for ds in datasets]
        self.data = da.concatenate(arrays, axis=0)

    def __getitem__(self, key):
        return self.data[key]
    
    def __repr__(self):
        return f"{self.reader} : {self.name} : run {self.run}"