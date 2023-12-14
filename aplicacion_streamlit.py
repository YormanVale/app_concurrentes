import streamlit as st
import os
from PIL import Image
from multiprocessing import Process, Queue, Pool
from descargar_imagenes import iniciar_descarga  # Asegúrate de tener esta función definida en "descargar_imagenes.py"
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.signal import convolve2d
import re


def apply_kernel_to_section(image_section, kernel):
    # Aplica el kernel a la sección de la imagen y retorna el resultado
    filtered_section = convolve2d(image_section, kernel, mode='same', boundary='wrap')
    return np.clip(filtered_section, 0, 255).astype(np.uint8)

def combine_sections(sections):
    # Combina las secciones procesadas en una sola imagen
    return np.vstack(sections)

def apply_kernel_multiprocessing(kernel, image_path, output_folder, num_processes=4):
    try:
        # Cargar y preparar la imagen
        image = Image.open(image_path)
        if image.mode != 'L':
            image = image.convert('L')

        image_array = np.array(image)
        if len(image_array.shape) != 2:
            raise ValueError('La imagen no es una imagen en escala de grises (2-D).')

        # Crear la carpeta de salida si no existe
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generar el nombre del archivo de salida
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        kernel_name = re.sub(r'[^a-zA-Z0-9]', '_', str(kernel))
        output_filename = f'{base_filename}_{kernel_name}_filtered.jpg'
        output_path = os.path.join(output_folder, output_filename)

        # Dividir la imagen en secciones para el multiprocessing
        height = image_array.shape[0]
        section_height = height // num_processes
        sections = [image_array[i:i+section_height] for i in range(0, height, section_height)]

        # Procesar cada sección en paralelo
        with Pool(num_processes) as p:
            processed_sections = p.starmap(apply_kernel_to_section, [(section, kernel) for section in sections])

        # Combinar las secciones procesadas en una imagen final
        filtered_image_array = combine_sections(processed_sections)

        # Guardar la imagen filtrada
        filtered_image = Image.fromarray(filtered_image_array, 'L')
        filtered_image.save(output_path)

        # Calcular estadísticas
        min_val, max_val, mean_val, std_val = filtered_image_array.min(), filtered_image_array.max(), filtered_image_array.mean(), filtered_image_array.std()

        return output_path, min_val, max_val, mean_val, std_val

    except Exception as e:
        print(f"Error al aplicar el kernel: {e}")
        return None

    

def get_kernels():
    kernels = {
        'kernel_class_1' : np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        'kernel_class_2' : np.array([[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]]),
        'kernel_class_3' : np.array([[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]]),
        'kernel_square_3x3' : np.ones((3, 3)),
        'kernel_edge_3x3' : np.array([[ 1,  0, -1],
                                      [ 0,  0,  0],
                                      [-1,  0,  1]]),
        'kernel_square_5x5' : np.ones((5, 5)),
        'kernel_edge_5x5' : np.array([[ 2,  1,  0, -1, -2],
                                      [ 1,  1,  0, -1, -1],
                                      [ 0,  0,  0,  0,  0],
                                      [-1, -1,  0,  1,  1],
                                      [-2, -1,  0,  1,  2]]),
        'kernel_sobel_vertical' : np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'kernel_sobel_horizontal' : np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        'kernel_laplace' : np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        'kernel_prewitt_vertical' : np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
        'kernel_prewitt_horizontal' : np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
        
    }
    return kernels


def main():
    st.title('Aplicación de Filtros de Imagen con Streamlit')

    # Sección para descargar imágenes
    st.subheader('Descargar Imágenes')
    api_key = 'BWyi6YIZiRbAOCKDH649CBPII8xVCgOzArRNv5wRivgyPIFpMYv98bz5'
    tema = 'computer'
    cantidad = 20
    
    def listar_imagenes(directory):
        return [f for f in listdir(directory) if isfile(join(directory, f))]

    n_hilos = 10

    if st.button('Descargar Imágenes'):
        print('Descargando imágenes...')
        iniciar_descarga(api_key, tema, cantidad, n_hilos)
        st.success('Descarga completada')

    imagenes = listar_imagenes('img')
    imagen_seleccionada = st.selectbox('Seleccione una imagen:', imagenes)

    if imagen_seleccionada:
        st.image(os.path.join('img', imagen_seleccionada), caption='Imagen Original')

    # Selección de filtro
    kernel_choice = st.selectbox('Seleccione el tipo de filtro:', list(get_kernels().keys()))
    
     # Sección para elegir el framework
    framework = st.radio("Selecciona el framework:", ("multiprocessing", "OpenMP"))


    if st.button('Aplicar filtro'):
        # Cargar la imagen seleccionada
        ruta_imagen = os.path.join('img', imagen_seleccionada)
        kernel = get_kernels()[kernel_choice]

        # Aplicar el filtro seleccionado
        if framework == "multiprocessing":
            result = apply_kernel_multiprocessing(kernel, ruta_imagen, 'output')
        elif framework == "OpenMP":
            # Agrega aquí la función correspondiente para OpenMP
            pass

        if result:  # Verifica si result tiene un valor asignado
            output_path, min_val, max_val, mean_val, std_val = result
            # Recuperar la imagen filtrada y mostrarla
            try:
                filtered_image = Image.open(output_path)
                st.image(filtered_image, caption='Imagen Filtrada')
                
                # Convertir la imagen a un array numpy
                filtered_image_array = np.array(filtered_image)

                
                # Calcular estadísticas
                min_val = filtered_image_array.min()
                max_val = filtered_image_array.max()
                mean_val = filtered_image_array.mean()
                std_val = filtered_image_array.std()
                
                # Mostrar estadísticas
                st.write(f"Valor mínimo: {min_val}")
                st.write(f"Valor máximo: {max_val}")
                st.write(f"Valor medio: {mean_val}")
                st.write(f"Desviación estándar: {std_val}")
                
                return output_path, min_val, max_val, mean_val, std_val

            except Exception as e:
                st.error(f"Error al abrir la imagen filtrada: {e}")
        else:
            st.error("No se pudo aplicar el filtro a la imagen.")

if __name__ == '__main__':
    main()
