import numpy as np
import cv2
import os

# Caminho alternativo e verificação de múltiplas localizações
possible_paths = [
    os.path.join('DETECCAO_FACIAL_BASICA_COM_OPENCV', 'haarcascade_frontalface_default.xml'),
    'haarcascade_frontalface_default.xml',  # Na pasta atual
    os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')  # Caminho padrão do OpenCV
]

faceCascade = None
for cascade_path in possible_paths:
    if os.path.isfile(cascade_path):
        faceCascade = cv2.CascadeClassifier(cascade_path)
        if not faceCascade.empty():
            print(f"Classificador carregado de: {cascade_path}")
            break

# Se nenhum classificador foi encontrado
if faceCascade is None or faceCascade.empty():
    print("ERRO: Classificador não encontrado em nenhum dos locais:")
    for path in possible_paths:
        print(f" - {path}")
    print("\nSoluções possíveis:")
    print("1. Baixe o arquivo do GitHub do OpenCV:")
    print("   https://github.com/opencv/opencv/tree/master/data/haarcascades")
    print("2. Coloque-o na pasta do script ou em 'DETECCAO_FACIAL_BASICA_COM_OPENCV'")
    print("3. Instale o OpenCV completo: pip install opencv-contrib-python")
    exit()

# Inicializar câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERRO: Não foi possível acessar a câmera!")
    exit()

cap.set(3, 640)  # Largura
cap.set(4, 480)  # Altura

# Configurações ajustáveis
SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 6
MIN_SIZE = (40, 40)
BRIGHTNESS_ADJ = 40
CONTRAST_ADJ = 1.2

print("\nDETECÇÃO FACIAL - CONTROLES:")
print("ESC: Sair")
print("+: Aumentar precisão")
print("-: Aumentar velocidade")
print("n: Reduzir falsos positivos")
print("m: Aumentar sensibilidade")

while True:
    ret, img = cap.read()
    if not ret:
        print("Erro: Frame não capturado")
        break
        
    
    # Pré-processamento
    img = cv2.convertScaleAbs(img, alpha=CONTRAST_ADJ, beta=BRIGHTNESS_ADJ)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Detecção facial
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Desenhar resultados
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, (x + w//2, y + h//2), 5, (0, 0, 255), -1)
        cv2.putText(img, f"Rosto", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Informações na tela
    cv2.putText(img, f"Faces: {len(faces)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(img, f"Config: SF={SCALE_FACTOR:.2f}, MN={MIN_NEIGHBORS}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
    cv2.putText(img, "Controles: + - n m", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,100,255), 1)
    
    cv2.imshow('Detecção Facial', img)

    # Controles de teclado
    k = cv2.waitKey(30)
    if k == 27:  # ESC
        break
    elif k == ord('+') and SCALE_FACTOR > 1.01:
        SCALE_FACTOR -= 0.01
    elif k == ord('-') and SCALE_FACTOR < 1.5:
        SCALE_FACTOR += 0.01
    elif k == ord('n') and MIN_NEIGHBORS < 10:
        MIN_NEIGHBORS += 1
    elif k == ord('m') and MIN_NEIGHBORS > 1:
        MIN_NEIGHBORS -= 1

cap.release()
cv2.destroyAllWindows()