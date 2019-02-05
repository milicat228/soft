import cv2
import numpy as np

def process_image(image, channel = 2, iterations = 1):
    """Funkcija namenjena za pretprocesiranje pojedinačnih frejmova."""
    #iz BGR formata se izdavaja samo prosleđeni kanal 
    image = image[:, :, channel]
    #blur se primenjuje da smanji velicinu tackica
    image = cv2.medianBlur(image,3)
    #otvaranje = erozija + dilacija (da bi se uklonile tackice)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.erode(image, kernel, iterations)
    image = cv2.dilate(image, kernel, iterations)
    #globalni treshold prevodi sliku u binarnu
    _, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)    
    return image

def process_region(region):
     """Funkcija namenjena za pretprocesiranje kontura pre ulaska u neuronsku mrežu."""
     #promena veličine regiona na 28x28 kao što neuronska mreža očekuje
     return cv2.resize(region,(28,28), interpolation = cv2.INTER_AREA)