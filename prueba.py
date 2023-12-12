import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
from imageio import imread
import matplotlib.pyplot as plt

# Cargar la imagen
image = imread('fig-3-2x.jpg', pilmode='L')  # Asumiendo que es en escala de grises

# Definir los kernels
kernel_class_1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
kernel_class_2 = np.array([[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]])
kernel_class_3 = np.array([[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]])
kernel_square_3x3 = np.ones((3, 3))  # Ejemplo de un kernel cuadrado 3x3
kernel_edge_3x3 = np.array([[ 1,  0, -1],
 [ 0,  0,  0],
 [-1,  0,  1]
])
kernel_square_5x5 = np.ones((5, 5))  # Ejemplo de un kernel cuadrado 5x5
kernel_edge_5x5 = np.array([[ 2,  1,  0, -1, -2],
 [ 1,  1,  0, -1, -1],
 [ 0,  0,  0,  0,  0],
 [-1, -1,  0,  1,  1],
 [-2, -1,  0,  1,  2]
])
kernel_sobel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_sobel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
kernel_laplace = ndimage.laplace(image)  # La función de scipy para Laplace
kernel_prewitt_vertical = ndimage.prewitt(image, axis=0)  # Prewitt vertical
kernel_prewitt_horizontal = ndimage.prewitt(image, axis=1)  # Prewitt horizontal

# Aplicar el kernel a la imagen
filtered_image_class_1 = convolve2d(image, kernel_class_1, mode='same', boundary='wrap')

# Calcular las estadísticas solicitadas
min_val = filtered_image_class_1.min()
max_val = filtered_image_class_1.max()
mean_val = filtered_image_class_1.mean()
std_dev = filtered_image_class_1.std()

# Mostrar las estadísticas
print(f"Dimensiones: {filtered_image_class_1.shape}")
print(f"Valor mínimo: {min_val}")
print(f"Valor máximo: {max_val}")
print(f"Valor medio: {mean_val}")
print(f"Desviación estándar: {std_dev}")

# Para visualizar la imagen filtrada (opcional)
plt.imshow(filtered_image_class_1, cmap='gray')
plt.show()
