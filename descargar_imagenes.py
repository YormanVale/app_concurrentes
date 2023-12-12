import requests
import threading
import os

def descargar_img(api_key, tema, cantidad, inicio, hilo_id):
    print(f"Hilo {hilo_id} empezando.")

    headers = {'Authorization': api_key}
    for i in range(inicio, inicio + cantidad):
        try:
            params = {'query': tema, 'per_page': 80, 'page': i}
            response = requests.get('https://api.pexels.com/v1/search', headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['photos']:
                    image_url = data['photos'][0]['src']['original']
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        with open(f'img/imagen_{i}_{hilo_id}.jpg', 'wb') as file:  # Cambio en la ruta
                            file.write(image_response.content)
                    else:
                        print(f"Hilo {hilo_id}: Error al descargar la imagen, código de estado {image_response.status_code}")
                else:
                    print(f"Hilo {hilo_id}: No se encontraron fotos en la página {i}")
            else:
                print(f"Hilo {hilo_id}: Error en la respuesta de la API, código de estado {response.status_code}")
        except Exception as e:
            print(f"Hilo {hilo_id}: Excepción al procesar la página {i}: {e}")

    print(f"Hilo {hilo_id} terminado.")



def iniciar_descarga(api_key, tema, cantidad, n_hilos):
    # Crear carpeta img si no existe
    if not os.path.exists('img'):
        os.makedirs('img')

    hilos = []
    for i in range(n_hilos):
        inicio = i * cantidad + 1
        hilo_id = i + 1
        hilo = threading.Thread(target=descargar_img, args=(api_key, tema, cantidad, inicio, hilo_id))
        hilos.append(hilo)
        hilo.start()

    for hilo in hilos:
        hilo.join()


n_hilos = 10
imagenes_por_hilo = 20
api_key = 'BWyi6YIZiRbAOCKDH649CBPII8xVCgOzArRNv5wRivgyPIFpMYv98bz5'
tema = 'computer'



