import scipy
import numpy as np
import cv2
import math
import image_utils as iu
import neural_network as nn
import numpy.linalg
from collections import OrderedDict
import random

class TrackingObject:
    """Predstavlja jednu konturu za koju se vrši praćenje."""
    network = nn.NeuralNetwork() #PAZI: Ovo je staticko za sve objekte
    blue_line = []
    green_line = []

    #detektovanje
    id = 0 #oznaka koju je tracker dodelio konturi
    times_detected = 0 #koliko puta je kontura bila detektovana
    lost_for = 0 #koliko frejmova je kontura izgubljena   

    #o konturi
    average_size = None #prosecna veličina u obliku w,h (sluzi za procenu izlaska van videa)
    first_location = np.array([0,0]) #x i y koordinate na kojima je kontura prvi put uocena
    first_frame_of_detection = 0 #indeks frejma na kome je prvi put detekovan objekat
    last_location = np.array([0,0]) #x i y koordinate na kojima je kontura uocena poslednji put    
    last_frame_of_detection = 0 #indeks frejma na kome je poslednji put detektovana

    #o broju
    last_prediction = -1 #na kom frejmu je rađena poslednja detekcija
    votes = None #verovatnoce da je neki broj
    times_predicted = 0

    #o presecima
    crossed_blue = 0
    crossed_green = 0

    def __init__(self, id, contour, frame_number, image):
        """Kreira objekat koji je prvi put pronađen."""
        self.id = id
        self.found(contour, frame_number, image)
        self.first_location = self.last_location
        self.first_frame_of_detection = frame_number
        self.crossed_blue = 0
        self.crossed_green = 0

    def found(self, contour, frame_number, image):
        """Beleži pronalazak konture u okviru frejma. Contour predstavlja konturu sa slike za koju se misli da je ova.""" 
         #izračunaj centar konture i sačuvaj poslednju poziciju i procesni pomeraj
        M = cv2.moments(contour) 
        self.last_location = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        #zabelezi neke osnovne informacije        
        self.times_detected += 1
        self.last_frame_of_detection = frame_number
        self.lost_for = 0
        #zabeleži veličinu
        _,_,w,h = cv2.boundingRect(contour) 
        if self.average_size is None:
            self.average_size = np.array([w, h])
        else:
            self.average_size = (self.average_size + np.array([w, h]))/2

        if self.last_prediction == -1: #prvi put kada je nađena uradi predikciju
            self.predict(contour, image)
        elif (self.last_frame_of_detection - self.last_prediction) > 15: #pravi pauze u detekcijama
            self.predict(contour, image)
        self.check_for_crossing()

    def lost(self):
        """Zabeleži da je cifra izgubljena."""
        self.lost_for += 1 
        self.check_for_crossing()

    def is_out_of_frame(self):
        """Proverava da li je objekat okončao svoj životni vek u videu."""
        #dimenzije videa su širina: 640, visina: 480
        #uzmi poziciju     
        aprox_position = self.get_position()

        if aprox_position[0] > 640 - self.average_size[0]: #x-osa
            return True
        
        if aprox_position[1] > 480 - self.average_size[1]: #y-osa
            return True

        return False

    def get_position(self):
        """Ako je objekat izgubljen, vraća aproksimaciju pozicije, inače vraća zadnju poziciju."""
        if self.lost_for == 0:
            return self.last_location
        else:
            #aproksimiraj koliko bi objekat prešao 
            #koliko dugo je putovao od prve do poslednje poznate lokacije
            number = self.last_frame_of_detection - self.first_frame_of_detection
            if number == 0:
                aprox_speed = np.array([0.5, 1]) #svi brojevi na videu se krecu u desno i na dole
            else:
                aprox_speed = (self.last_location - self.first_location)/number 
            return self.last_location + aprox_speed * self.lost_for
    
    def position_distance(self, contour):
        """Pronalazi udaljenost između dve pojave na frejmovima. Koristi euklidsku udaljenost centara kontura."""
        # pronađi centar konture
        M = cv2.moments(contour)
        position = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        #uzmi poziciju     
        self_position = self.get_position()
        #izračunaj euklidsku udaljenost
        dist = math.sqrt( (position[0] - self_position [0])**2 + (position[1] - self_position[1])**2 )
        return dist        

    def distance(self, contour):
        """Pronalazi udaljenost između dve pojave na frejmovima."""
        # ideje: uzeti velicinu o obzir, uzeti broj koji je neuronska mreža pronašla u obzir        
        return self.position_distance(contour)

    def predict(self, contour, image):
        """Vrsi detekciju broja sa konture"""
        self.last_prediction = self.last_frame_of_detection
        self.times_predicted += 1
        #iseci konturu sa slike
        x,y,w,h = cv2.boundingRect(contour) 
        region = image[y:y+h+1,x:x+w+1]
        region = iu.process_region(region)
        #pošalji neuronskoj mreži da izvrši detekciju
        prediction = self.network.predict([region]) 
        if self.votes is not None:        
            self.votes = self.votes + prediction  
        else:
            self.votes = prediction

    def value(self): 
        """Vraca koji broj je detektovan na objektu."""
        return max(enumerate(self.votes[0]), key=lambda x: x[1])[0]

    def ccw(self,A,B,C):
        return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

    def intersect(self,A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)
    
    def check_for_crossing(self):
        """Proverava da li je presecena linija."""
        pos = self.get_position()
        C = Pos(self.first_location[0], self.first_location[1])
        D = Pos(pos[0], pos[1])
        if self.crossed_blue < 50:
            A = Pos(self.blue_line[0], self.blue_line[1])
            B = Pos(self.blue_line[2], self.blue_line[3])            
            if self.intersect(A,B,C,D):
                self.crossed_blue += 1
        if self.crossed_green < 50:
            A = Pos(self.green_line[0], self.green_line[1])
            B = Pos(self.green_line[2], self.green_line[3])            
            if self.intersect(A,B,C,D):
                self.crossed_green += 1

   

class Pos:
    def __init__(self,x,y):
        self.x = x
        self.y = y

     

class Tracker:
    """Klasa zadužena da čuva sve objekte uočene na videu."""
    next_id = 0 #oznake za objekata (služi da bi se novopronađenom objektu dodelila oznaka)
    objects = OrderedDict() #hashmapa objekata za koje se zna pozicija
    lost_objects = OrderedDict() #hashmapa objekata koji su izgubljeni
    out_of_frame_objects = OrderedDict() #hashmapa objekata koji su izašli iz okvira (stari objekti)

    def __init__(self, blue_line, green_line):
        self.next_id = 0 
        self.objects = OrderedDict() 
        self.lost_objects = OrderedDict() 
        self.out_of_frame_objects = OrderedDict() 
        TrackingObject.blue_line = blue_line
        TrackingObject.green_line = green_line

    def process_frame(self, contours, frame_number, image):
         """Procesira uočene konture brojeva na jednom frejmu."""
         #print('Working on frame: ' + str(frame_number))
         #ako nije pronađena ni jedna kontura, javi svim objektima da su izgubljeni 
         if len(contours) == 0:
             for key, value in self.objects:
                 self.lost_objects[key] = value
                 value.lost()
             for key, value in self.lost_objects:
                 value.lost()
             #isprazni nađene
             self.objects = OrderedDict()
             #proveri da li su neki objekti napustili video
             self.check_for_out_of_frame_objects(self.lost_objects)

         found_in_this_frame = OrderedDict()
         #glavni deo - analiza kontura
         for contour in contours:
             #analiza sa objektima koji nisu izgubljeni
             if len(self.objects): #proveri da ima barem jedan registovani
                min_dist_from_objects = self.find_min_distance(contour, self.objects)
                if min_dist_from_objects['distance'] < 4:
                    #obj = min_dist_from_objects['object']
                    #print('Found again: ' + str(obj.id) + ' with value' + str(obj.value()) )
                    #pronađen je postojeci
                    min_dist_from_objects['object'].found(contour, frame_number,image)
                    found_in_this_frame[min_dist_from_objects['object'].id] = min_dist_from_objects['object']
                    #izbaci objekat iz object mape, jer ne moze jos neko da bude on
                    del self.objects[min_dist_from_objects['object'].id]
                    continue

             #pokušaj da ga nađeš među izgubljenim
             if len(self.lost_objects):
                min_dist_from_lost = self.find_min_distance(contour, self.lost_objects)             
                if min_dist_from_lost['distance'] < 20: 
                    #obj = min_dist_from_lost['object']
                    #print('Found lost: ' + str(obj.id) + ' with value' + str(obj.value()) )
                    min_dist_from_lost['object'].found(contour, frame_number, image)
                    found_in_this_frame[min_dist_from_lost['object'].id] = min_dist_from_lost['object']
                    #izbaci objekat iz object mape, jer ne moze jos neko da bude on
                    del self.lost_objects[min_dist_from_lost['object'].id]
                    continue                    

             #Ako je kontura previse blizu ivicama, nemoj kreirati novu
             #Potrebno radi tacne detekcije izlaska 
             M = cv2.moments(contour) 
             center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]]) 
             space = 17
             if center[0] < space or center[0] > (640 - space):
                 continue
             if center[1] < space or center[1] > (480 - space):
                 continue
             #ovde su nove konture
             obj = TrackingObject(self.next_id, contour, frame_number, image)
             found_in_this_frame[self.next_id] = obj
             #print('Created: ' + str(obj.id) + ' with value' + str(obj.value()) )
             self.next_id += 1


         #popravi liste
         #svi koji su ostali u objects su zapravo lost
         self.lost_objects.update(self.objects)
         #javi izgubljenim da su bili jos jedan frejm izgubljeni
         for key,value in self.lost_objects.items():
             #print('Lost: ' + str(value.id) + ' with value' + str(value.value()) )
             value.lost()
         #u objects listu se dodaju oni koji su bili na frejmu
         self.objects = found_in_this_frame

         #proveri obe liste u potrazi za objektima koji su napustili video
         #print('Checking found objects for out of frame')
         self.check_for_out_of_frame_objects(self.objects)
         #print('Checking lost objects for out of frame')
         self.check_for_out_of_frame_objects(self.lost_objects)


    def find_min_distance(self, contour, objects):
         """Prolazi objekat u listi najbliži konturi."""
         min_obj = next(iter(objects.values()))
         min_dist = min_obj.distance(contour)
         for _, value in objects.items():
            dist = value.distance(contour)
            if dist < min_dist:
                min_dist = dist
                min_obj = value

         return {'object': min_obj, 'distance': min_dist}

    def check_for_out_of_frame_objects(self, objects):
         """Prolazi kroz listu objekata i arhivira sve objekte koji su izašli van okvira frejma."""
         ids = []
         for key, value in objects.items():
            if value.is_out_of_frame():
                ids.append(key)
                self.out_of_frame_objects[key] = value
                #print('Out of frame ' + str(key) + ' value ' + str(value.value()) )
         for id in ids:
            del objects[id]
    
    def draw_all_traces(self, image):
         self.draw_traces_of_objects(image, self.objects, (0,0,255))
         self.draw_traces_of_objects(image, self.lost_objects, (0,255,0))
         #self.draw_traces_of_objects(image, self.out_of_frame_objects, (255,0,0))

    def draw_traces_of_objects(self, image, objects, color):
        for _, value in objects.items():
             start = value.first_location
             end = value.get_position()
             cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, 1, cv2.LINE_AA) 


    def line_sum(self, param):
        A = self.sum_per_list(param, self.objects)
        B = self.sum_per_list(param, self.lost_objects)
        C = self.sum_per_list(param, self.out_of_frame_objects)
        return  A + B + C

    def sum_per_list(self, param, objects):        
        sum = 0
        for key, value in objects.items():   
            if param == 0 and value.crossed_blue >= 5:
                #print(value.value())
                sum += value.value()
            elif param == 1 and value.crossed_green >= 15:
                sum += value.value()
        return sum

    