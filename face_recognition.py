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
        
        # Convertir frame a formato compatible con AWS Rekognition
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        try:
            logger.info("Enviando imagen a AWS Rekognition para búsqueda de rostros...")
            # Buscar coincidencias en la colección
            response = self.rekognition.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={'Bytes': img_bytes},
                MaxFaces=10,  # Aumentado para detectar más rostros
                FaceMatchThreshold=self.similarity_threshold
            )
            
            # Verificar si hay rostros detectados por AWS
            if 'FaceMatches' not in response or 'SearchedFaceBoundingBox' not in response:
                logger.warning("AWS no detectó ningún rostro en la imagen")
                # Devolver rostros no reconocidos
                return [(x, y, w, h, False, "No autorizado") for (x, y, w, h) in faces_opencv]
            
            # Obtener información de los rostros detectados por AWS
            face_matches = response.get('FaceMatches', [])
            aws_face_box = response.get('SearchedFaceBoundingBox', {})
            
            # Convertir coordenadas relativas de AWS a píxeles
            aws_x = int(aws_face_box.get('Left', 0) * frame.shape[1])
            aws_y = int(aws_face_box.get('Top', 0) * frame.shape[0])
            aws_w = int(aws_face_box.get('Width', 0) * frame.shape[1])
            aws_h = int(aws_face_box.get('Height', 0) * frame.shape[0])
            
            logger.info(f"Respuesta de AWS: {len(face_matches)} coincidencias encontradas")
            logger.info(f"AWS detectó rostro en: ({aws_x}, {aws_y}, {aws_w}, {aws_h})")
            
            # Si no hay coincidencias, devolver rostros no reconocidos
            if len(face_matches) == 0:
                logger.warning("No se encontraron coincidencias en la colección")
                return [(x, y, w, h, False, "No autorizado") for (x, y, w, h) in faces_opencv]
            
            # Encontrar el rostro de OpenCV que mejor coincide con el rostro detectado por AWS
            best_match_index = -1
            best_iou = 0
            
            for i, (x, y, w, h) in enumerate(faces_opencv):
                # Calcular IoU (Intersection over Union)
                x_overlap = max(0, min(x+w, aws_x+aws_w) - max(x, aws_x))
                y_overlap = max(0, min(y+h, aws_y+aws_h) - max(y, aws_y))
                overlap_area = x_overlap * y_overlap
                area1 = w * h
                area2 = aws_w * aws_h
                
                if area1 + area2 - overlap_area > 0:  # Evitar división por cero
                    iou = overlap_area / float(area1 + area2 - overlap_area)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match_index = i
            
            # Procesar cada coincidencia encontrada
            results = []
            
            # Si encontramos una buena coincidencia entre AWS y OpenCV
            if best_match_index >= 0 and best_iou > 0.3:  # Umbral de IoU
                # Obtener el rostro de OpenCV que mejor coincide con AWS
                x, y, w, h = faces_opencv[best_match_index]
                
                # Obtener la mejor coincidencia de AWS
                best_aws_match = face_matches[0]  # La primera coincidencia es la mejor
                face = best_aws_match['Face']
                external_id = face.get('ExternalImageId', 'Desconocido')
                similarity = best_aws_match['Similarity']
                
                logger.info(f"Rostro reconocido: {external_id} (Similitud: {similarity:.2f}%, IoU: {best_iou:.2f})")
                results.append((x, y, w, h, True, f"{external_id} ({similarity:.1f}%)"))
                
                # Marcar este rostro como procesado
                processed_faces = [best_match_index]
            else:
                processed_faces = []
                logger.warning(f"No se encontró buena coincidencia entre AWS y OpenCV (mejor IoU: {best_iou:.2f})")
            
            # Añadir rostros no reconocidos
            for i, (x, y, w, h) in enumerate(faces_opencv):
                if i not in processed_faces:
                    results.append((x, y, w, h, False, "No autorizado"))
            
            return results
                
        except self.rekognition.exceptions.InvalidParameterException:
            # No se detectaron rostros en la imagen
            logger.warning("AWS no pudo procesar la imagen - No se detectaron rostros")
            # Devolver rostros no reconocidos
            return [(x, y, w, h, False, "No autorizado") for (x, y, w, h) in faces_opencv]
        except Exception as e:
            logger.error(f"Error al comparar rostros: {e}")
            # Devolver rostros no reconocidos en caso de error
            return [(x, y, w, h, False, "Error") for (x, y, w, h) in faces_opencv]
    
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