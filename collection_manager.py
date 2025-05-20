import boto3
import os
import logging
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, COLLECTION_ID

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollectionManager:
    def __init__(self):
        # Inicializar cliente de Rekognition
        self.rekognition = boto3.client(
            'rekognition',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        self.collection_id = COLLECTION_ID
    
    def create_collection(self):
        """Crear una colección de rostros en AWS Rekognition"""
        try:
            # Verificar si la colección ya existe
            self.rekognition.describe_collection(CollectionId=self.collection_id)
            logger.info(f"La colección {self.collection_id} ya existe.")
            return True
        except self.rekognition.exceptions.ResourceNotFoundException:
            # Crear la colección si no existe
            response = self.rekognition.create_collection(CollectionId=self.collection_id)
            logger.info(f"Colección {self.collection_id} creada: {response['StatusCode']}")
            return True
        except Exception as e:
            logger.error(f"Error al crear la colección: {e}")
            return False
    
    def delete_collection(self):
        """Eliminar una colección de rostros"""
        try:
            response = self.rekognition.delete_collection(CollectionId=self.collection_id)
            logger.info(f"Colección {self.collection_id} eliminada: {response['StatusCode']}")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar la colección: {e}")
            return False
    
    def add_face_to_collection(self, image_path, external_id):
        """
        Añadir un rostro a la colección
        
        Args:
            image_path (str): Ruta a la imagen con el rostro
            external_id (str): ID externo para identificar a la persona (nombre, ID empleado, etc.)
        
        Returns:
            bool: True si se añadió correctamente, False en caso contrario
        """
        try:
            with open(image_path, 'rb') as image:
                response = self.rekognition.index_faces(
                    CollectionId=self.collection_id,
                    Image={'Bytes': image.read()},
                    ExternalImageId=external_id,
                    MaxFaces=1,
                    QualityFilter="AUTO",
                    DetectionAttributes=['ALL']
                )
                
                if len(response['FaceRecords']) > 0:
                    face_id = response['FaceRecords'][0]['Face']['FaceId']
                    logger.info(f"Rostro añadido a la colección. FaceId: {face_id}, ExternalImageId: {external_id}")
                    return True
                else:
                    logger.warning(f"No se detectó ningún rostro en la imagen {image_path}")
                    return False
                
        except Exception as e:
            logger.error(f"Error al añadir rostro a la colección: {e}")
            return False
    
    def list_faces(self):
        """Listar todos los rostros en la colección"""
        try:
            faces = []
            response = self.rekognition.list_faces(CollectionId=self.collection_id, MaxResults=100)
            faces.extend(response['Faces'])
            
            # Manejar paginación si hay más de 100 rostros
            while 'NextToken' in response:
                response = self.rekognition.list_faces(
                    CollectionId=self.collection_id,
                    MaxResults=100,
                    NextToken=response['NextToken']
                )
                faces.extend(response['Faces'])
            
            logger.info(f"Se encontraron {len(faces)} rostros en la colección")
            return faces
        except Exception as e:
            logger.error(f"Error al listar rostros: {e}")
            return []
    
    def delete_face(self, face_id):
        """Eliminar un rostro de la colección por su FaceId"""
        try:
            response = self.rekognition.delete_faces(
                CollectionId=self.collection_id,
                FaceIds=[face_id]
            )
            logger.info(f"Rostro eliminado: {face_id}")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar rostro: {e}")
            return False