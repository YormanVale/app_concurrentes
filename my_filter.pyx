# cython: language_level=3
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

cdef extern from "my_filter.c":
    void apply_filter_openmp(float *input, float *output, int width, int height)

def apply_filter(image_array, int width, int height):
    # Asegúrate de que image_array es un array de tipo float
    cdef cnp.ndarray[cnp.float32_t, ndim=2] c_input = image_array.astype(np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] c_output = np.empty([height, width], dtype=np.float32)

    apply_filter_openmp(<float*>c_input.data, <float*>c_output.data, width, height)

    return c_output
