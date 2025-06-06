import numpy as np
import cv2

# Carregar classificador (usando caminho absoluto mais confiável)
cascade_path = 'C:/Users/EDUARDO_VASCONCELOS/Downloads/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

# Verificar se o classificador foi carregado corretamente
if faceCascade.empty():
    print("Erro: Classificador não carregado!")
    exit()

# Inicializar câmera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Largura
cap.set(4, 480)  # Altura

# Configurações ajustáveis (experimente diferentes valores)
SCALE_FACTOR = 1.05      # Reduzido para maior precisão
MIN_NEIGHBORS = 6        # Aumentado para reduzir falsos positivos
MIN_SIZE = (40, 40)      # Aumentado para ignorar objetos pequenos
BRIGHTNESS_ADJ = 40      # Ajuste de brilho
CONTRAST_ADJ = 1.2       # Ajuste de contraste

while True:
    ret, img = cap.read()
    if not ret:
        break
        
    # Rotação única de 180°
    #img = cv2.rotate(img, cv2.ROTATE_180)
    
    # Pré-processamento para melhorar detecção
    img = cv2.convertScaleAbs(img, alpha=CONTRAST_ADJ, beta=BRIGHTNESS_ADJ)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Equalização de histograma
    gray = cv2.GaussianBlur(gray, (5,5), 0)  # Redução de ruído
    
    # Detecção facial com parâmetros otimizados
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Desenhar retângulos e informações
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Exibir precisão (tamanho relativo)
        face_percent = (w * h) / (img.shape[0] * img.shape[1]) * 100
        cv2.putText(img, f"{face_percent:.1f}%", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Adicionar FPS
    cv2.putText(img, f"Faces: {len(faces)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    cv2.imshow('Detecção Facial Aprimorada', img)

    # Tecla ESC para sair
    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()