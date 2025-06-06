import numpy as np
import cv2
import os

# Caminho corrigido para o classificador
cascade_path = os.path.join('DETECCAO_FACIAL_BASICA_COM_OPENCV', 'haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cascade_path)

# Verificar se o classificador foi carregado corretamente
if faceCascade.empty():
    print(f"Erro: Classificador não encontrado em {cascade_path}!")
    print("Verifique se o caminho está correto e o arquivo existe.")
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

print("Pressione ESC para sair...")
print("Dicas para melhor precisão:")
print("- Garanta boa iluminação no ambiente")
print("- Posicione o rosto a ~50cm da câmera")
print("- Evite fundos muito complexos")

while True:
    ret, img = cap.read()
    if not ret:
        print("Erro: Não foi possível capturar frame da câmera!")
        break
        
    # Rotação única de 180°
    img = cv2.rotate(img, cv2.ROTATE_180)
    
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
        # Retângulo principal
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Marcador de centro
        center_x = x + w//2
        center_y = y + h//2
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Exibir informações
        face_percent = (w * h) / (img.shape[0] * img.shape[1]) * 100
        cv2.putText(img, f"Face: {face_percent:.1f}%", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img, f"Pos: ({center_x},{center_y})", (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

    # Adicionar informações gerais
    cv2.putText(img, f"Faces Detectadas: {len(faces)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(img, f"Config: SF={SCALE_FACTOR}, MN={MIN_NEIGHBORS}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
    
    cv2.imshow('Detecção Facial Aprimorada', img)

    # Teclas de controle
    k = cv2.waitKey(30)
    if k == 27:  # ESC para sair
        break
    elif k == ord('+') and SCALE_FACTOR > 1.01:
        SCALE_FACTOR -= 0.01  # Aumentar precisão
    elif k == ord('-') and SCALE_FACTOR < 1.5:
        SCALE_FACTOR += 0.01  # Aumentar velocidade
    elif k == ord('n') and MIN_NEIGHBORS < 10:
        MIN_NEIGHBORS += 1   # Reduzir falsos positivos
    elif k == ord('m') and MIN_NEIGHBORS > 1:
        MIN_NEIGHBORS -= 1   # Aumentar sensibilidade

cap.release()
cv2.destroyAllWindows()