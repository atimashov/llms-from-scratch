cdef class CoocCounter:
    cdef dict data

    def __cinit__(self):
        self.data = {}

    cpdef void update(self, int i, int j, double value):
        cdef tuple key = (i, j)
        self.data[key] = self.data.get(key, 0.0) + value

    cpdef double get(self, int i, int j):
        return self.data.get((i, j), 0.0)

    def __len__(self):
        return len(self.data)

    def items(self):
        return self.data.items()