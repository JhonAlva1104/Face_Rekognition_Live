import cv2
import time
import argparse
import logging
import threading
import queue
from collection_manager import CollectionManager
from face_recognition import FaceRecognition

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cola para comunicación entre hilos
frame_queue = queue.Queue(maxsize=2)  # Aumentado a 2 para mejor rendimiento
result_queue = queue.Queue(maxsize=2)  # Aumentado a 2 para mejor rendimiento

# Función para el hilo de procesamiento
def process_frames(face_recognition):
    logger.info("Iniciando hilo de procesamiento de frames")
    while True:
        try:
            # Obtener el frame de la cola
            frame = frame_queue.get()
            
            # Si recibimos None, es señal para terminar
            if frame is None:
                logger.info("Terminando hilo de procesamiento")
                break
            
            # Procesar el frame
            processed_frame = face_recognition.process_frame(frame)
            
            # Poner el resultado en la cola de resultados
            # Si la cola está llena, descartar el frame anterior
            if result_queue.full():
                result_queue.get()
            result_queue.put(processed_frame)
            
            # Marcar la tarea como completada
            frame_queue.task_done()
        except Exception as e:
            logger.error(f"Error en el hilo de procesamiento: {e}")
            # Continuar con el siguiente frame en caso de error
            continue

def main():
    parser = argparse.ArgumentParser(description='Sistema de reconocimiento facial con AWS Rekognition')
    parser.add_argument('--mode', type=str, default='recognize', choices=['recognize', 'add', 'list', 'delete'],
                        help='Modo de operación: recognize, add, list, delete')
    parser.add_argument('--image', type=str, help='Ruta de la imagen para añadir a la colección')
    parser.add_argument('--name', type=str, help='Nombre o ID de la persona para añadir a la colección')
    parser.add_argument('--face-id', type=str, help='ID del rostro para eliminar de la colección')
    
    args = parser.parse_args()
    
    # Inicializar gestor de colección
    collection_manager = CollectionManager()
    
    # Asegurarse de que la colección existe
    collection_manager.create_collection()
    
    if args.mode == 'add':
        # Añadir rostro a la colección
        if not args.image or not args.name:
            logger.error("Se requiere --image y --name para añadir un rostro")
            return
        
        success = collection_manager.add_face_to_collection(args.image, args.name)
        if success:
            logger.info(f"Rostro de {args.name} añadido correctamente")
        else:
            logger.error(f"Error al añadir el rostro de {args.name}")
    
    elif args.mode == 'list':
        # Listar rostros en la colección
        faces = collection_manager.list_faces()
        print("\nRostros en la colección:")
        for i, face in enumerate(faces, 1):
            print(f"{i}. ID: {face['FaceId']}, Persona: {face.get('ExternalImageId', 'Desconocido')}")
    
    elif args.mode == 'delete':
        # Eliminar rostro de la colección
        if not args.face_id:
            logger.error("Se requiere --face-id para eliminar un rostro")
            return
        
        success = collection_manager.delete_face(args.face_id)
        if success:
            logger.info(f"Rostro con ID {args.face_id} eliminado correctamente")
        else:
            logger.error(f"Error al eliminar el rostro con ID {args.face_id}")
    
    elif args.mode == 'recognize':
        # Modo de reconocimiento en tiempo real
        face_recognition = FaceRecognition()
        
        # Iniciar captura de video
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("No se pudo acceder a la cámara")
            return
        
        logger.info("Iniciando reconocimiento facial en tiempo real. Presiona 'q' para salir.")
        
        # Iniciar hilo de procesamiento
        processing_thread = threading.Thread(target=process_frames, args=(face_recognition,))
        processing_thread.daemon = True
        processing_thread.start()
        
        # En la función main(), dentro del modo 'recognize'
        # Variables para controlar la frecuencia de envío de frames al hilo de procesamiento
        last_frame_time = 0
        frame_interval = 0.05  # Ajustado para mejor equilibrio entre rendimiento y fluidez
        
        # Variable para almacenar el último frame procesado
        last_processed_frame = None
        last_raw_frame = None
        
        # Contador de FPS
        fps_counter = 0
        fps_timer = time.time()
        fps = 0
        
        while True:
            # Capturar frame
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Error al capturar frame de la cámara")
                break
            
            # Guardar una copia del frame actual
            last_raw_frame = frame.copy()
            
            current_time = time.time()
            
            # Calcular FPS
            fps_counter += 1
            if current_time - fps_timer >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_timer = current_time
            
            # Enviar frame al hilo de procesamiento con una frecuencia controlada
            if current_time - last_frame_time >= frame_interval and not frame_queue.full():
                # Hacer una copia del frame para el hilo
                frame_copy = frame.copy()
                
                # Poner el frame en la cola para procesamiento
                frame_queue.put(frame_copy)
                last_frame_time = current_time
            
            # Verificar si hay un nuevo frame procesado disponible
            if not result_queue.empty():
                last_processed_frame = result_queue.get()
            
            # Mostrar el último frame procesado si está disponible, o el frame sin procesar
            display_frame = last_processed_frame if last_processed_frame is not None else last_raw_frame
            
            # Añadir información de FPS
            cv2.putText(display_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar resultado
            cv2.imshow('Reconocimiento Facial', display_frame)
            
            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Señalizar al hilo de procesamiento que termine
        frame_queue.put(None)
        
        # Esperar a que el hilo termine
        processing_thread.join(timeout=1.0)
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()