import boto3
import cv2
import numpy as np
import logging
from PIL import Image
from io import BytesIO
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, COLLECTION_ID, SIMILARITY_THRESHOLD

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognition:
    def __init__(self):
        # Inicializar cliente de Rekognition
        self.rekognition = boto3.client(
            'rekognition',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        self.collection_id = COLLECTION_ID
        self.similarity_threshold = SIMILARITY_THRESHOLD
        
        # Inicializar detector de rostros de OpenCV para detección local
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("FaceRecognition inicializado correctamente")
    
    def detect_faces_opencv(self, frame):
        """
        Detectar rostros en un frame usando OpenCV (más rápido para detección local)
        
        Args:
            frame: Frame de video capturado por OpenCV
            
        Returns:
            list: Lista de coordenadas de rostros detectados [(x, y, w, h), ...]
        """
        # Reducir el tamaño del frame para acelerar el procesamiento
        scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Usar un detector más avanzado si está disponible
        try:
            # Intentar usar el detector DNN (más preciso)
            net = cv2.dnn.readNetFromCaffe(
                "models/deploy.prototxt",
                "models/res10_300x300_ssd_iter_140000.caffemodel"
            )
            
            blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            
            faces_original_size = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Umbral de confianza
                    box = detections[0, 0, i, 3:7] * np.array([small_frame.shape[1], small_frame.shape[0], small_frame.shape[1], small_frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Convertir a formato (x, y, w, h) y escalar al tamaño original
                    x_orig = int(startX / scale_factor)
                    y_orig = int(startY / scale_factor)
                    w_orig = int((endX - startX) / scale_factor)
                    h_orig = int((endY - startY) / scale_factor)
                    
                    faces_original_size.append((x_orig, y_orig, w_orig, h_orig))
            
            if len(faces_original_size) > 0:
                logger.info(f"DNN detectó {len(faces_original_size)} rostros")
                return faces_original_size
                
        except Exception as e:
            logger.warning(f"Error al usar detector DNN: {e}. Usando Haar Cascade como respaldo.")
        
        # Usar Haar Cascade como respaldo
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,  # Reducido para detectar más rostros
            minSize=(20, 20),  # Tamaño mínimo reducido
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Ajustar las coordenadas al tamaño original del frame
        faces_original_size = []
        for (x, y, w, h) in faces:
            x_orig = int(x / scale_factor)
            y_orig = int(y / scale_factor)
            w_orig = int(w / scale_factor)
            h_orig = int(h / scale_factor)
            faces_original_size.append((x_orig, y_orig, w_orig, h_orig))
        
        if len(faces_original_size) > 0:
            logger.info(f"OpenCV detectó {len(faces_original_size)} rostros")
        
        return faces_original_size
    
    def compare_faces_with_collection(self, frame):
        """
        Comparar rostros detectados en el frame con la colección de AWS
        
        Args:
            frame: Frame de video capturado por OpenCV
            
        Returns:
            list: Lista de tuplas (coordenadas, reconocido, nombre)
                 [(x, y, w, h, reconocido, nombre), ...]
        """
        # Detectar rostros con OpenCV primero
        faces_opencv = self.detect_faces_opencv(frame)
        
        if len(faces_opencv) == 0:
            return []
        
        # Lista para almacenar los resultados
        results = []
        
        # Procesar cada rostro detectado por OpenCV individualmente
        for i, (x, y, w, h) in enumerate(faces_opencv):
            try:
                # Recortar el rostro del frame
                face_img = frame[y:y+h, x:x+w]
                
                # Verificar que la imagen recortada no esté vacía
                if face_img.size == 0:
                    logger.warning(f"Imagen de rostro vacía para el rostro {i}")
                    results.append((x, y, w, h, False, "No autorizado"))
                    continue
                    
                # Convertir rostro recortado a formato compatible con AWS Rekognition
                _, img_encoded = cv2.imencode('.jpg', face_img)
                img_bytes = img_encoded.tobytes()
                
                logger.info(f"Enviando rostro {i} a AWS Rekognition para búsqueda...")
                
                # Buscar coincidencias en la colección para este rostro específico
                response = self.rekognition.search_faces_by_image(
                    CollectionId=self.collection_id,
                    Image={'Bytes': img_bytes},
                    MaxFaces=1,  # Solo necesitamos la mejor coincidencia para este rostro
                    FaceMatchThreshold=self.similarity_threshold
                )
                
                # Verificar si hay coincidencias para este rostro
                if 'FaceMatches' in response and len(response['FaceMatches']) > 0:
                    # Obtener la mejor coincidencia
                    best_match = response['FaceMatches'][0]
                    face = best_match['Face']
                    external_id = face.get('ExternalImageId', 'Desconocido')
                    similarity = best_match['Similarity']
                    
                    logger.info(f"Rostro {i} reconocido: {external_id} (Similitud: {similarity:.2f}%)")
                    results.append((x, y, w, h, True, f"{external_id} ({similarity:.1f}%)"))
                else:
                    # No se encontraron coincidencias para este rostro
                    logger.info(f"Rostro {i} no reconocido")
                    results.append((x, y, w, h, False, "No autorizado"))
                    
            except self.rekognition.exceptions.InvalidParameterException:
                # AWS no pudo procesar este rostro
                logger.warning(f"AWS no pudo procesar el rostro {i}")
                results.append((x, y, w, h, False, "No autorizado"))
            except Exception as e:
                # Error general
                logger.error(f"Error al procesar el rostro {i}: {e}")
                results.append((x, y, w, h, False, "Error"))
        
        return results
                
    def process_frame(self, frame):
        """
        Procesar un frame de video para detectar y reconocer rostros
        
        Args:
            frame: Frame de video capturado por OpenCV
            
        Returns:
            frame: Frame con anotaciones de reconocimiento
        """
        # Hacer una copia del frame para no modificar el original
        frame_copy = frame.copy()
        
        # Obtener resultados del reconocimiento facial
        recognized_faces = self.compare_faces_with_collection(frame)
        
        # Si no hay rostros detectados, devolver el frame sin modificar
        if len(recognized_faces) == 0:
            return frame_copy
        
        # Dibujar recuadros para cada rostro según su estado de reconocimiento
        for face_info in recognized_faces:
            x, y, w, h, recognized, name = face_info
            
            # Determinar el color según si está reconocido o no
            if recognized:
                # Color verde para rostros reconocidos
                color = (0, 255, 0)
            else:
                # Color rojo para rostros no reconocidos
                color = (0, 0, 255)
            
            # Dibujar recuadro
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, 2)
            
            # Añadir etiqueta con el nombre
            cv2.putText(frame_copy, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            logger.info(f"Dibujando recuadro {'verde' if recognized else 'rojo'} para {name} en ({x}, {y}, {w}, {h})")
        
        return frame_copy