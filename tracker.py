import scipy
import numpy as np
import cv2
import math
import image_utils as iu
import neural_network as nn
from collections import OrderedDict

class TrackingObject:
    """Predstavlja jednu konturu za koju se vrši praćenje."""
    network = nn.NeuralNetwork() #PAZI: Ovo je staticko za sve objekte
    #detektovanje
    id = 0 #oznaka koju je tracker dodelio konturi
    times_detected = 0 #koliko puta je kontura bila detektovana
    lost_for = 0 #koliko frejmova je kontura izgubljena
    first_location = [] #x i y koordinate na kojima je kontura prvi put uocena
    first_frame_of_detection = 0 #indeks frejma na kome je prvi put detekovan objekat
    last_location = [] #x i y koordinate na kojima je kontura uocena poslednji put    
    last_frame_of_detection = 0 #indeks frejma na kome je poslednji put detektovana

    #o konturi
    average_size = np.array([0,0]) #prosecna veličina u obliku w,h
    votes = np.zeros(10) #verovatnoce da je neki broj

    def __init__(self, id, contour, frame_number, image):
        """Kreira objekat koji je prvi put pronađen."""
        self.id = id
        self.found(contour, frame_number, image)
        self.first_location = self.last_location
        self.first_frame_of_detection = frame_number

    def found(self, contour, frame_number, image):
        """Beleži pronalazak konture u okviru frejma. Contour predstavlja konturu sa slike za koju se misli da je ova.""" 
        self.times_detected += 1
        self.last_frame_of_detection = frame_number
        self.lost_for = 0
        #zabeleži veličinu
        _,_,w,h = cv2.boundingRect(contour)        
        self.average_size = (self.average_size +  np.array([w, h]))/self.times_detected
        self.predict(contour, image)
        #izračunaj centar konture i sačuvaj poslednju poziciju
        M = cv2.moments(contour) 
        self.last_location = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]]) 

    def lost(self):
        """Zabeleži da je cifra izgubljena."""
        self.lost_for += 1

    def is_out_of_frame(self):
        """Proverava da li je objekat okončao svoj životni vek u videu."""
        #dimenzije videa su širina: 640, visina: 480
        space = 20 #ne mora centar da bude na ivici, dovoljno da je blizu
        #uzmi poziciju     
        aprox_position = self.aprox_position()

        if aprox_position[0] > 640 - space: #x-osa
            return True
        
        if aprox_position[1] > 480 - space: #y-osa
            return True

        return False

    def aprox_position(self):
        """Ako je objekat izgubljen, vraća aproksimaciju pozicije, inače vraća zadnju poziciju."""
        if self.lost_for == 0:
            return self.last_location
        else:
            #aproksimiraj koliko bi objekat prešao 
            #koliko dugo je putovao od prve do poslednje poznate lokacije
            number = self.last_frame_of_detection - self.first_frame_of_detection
            if number == 0:
                aprox_speed = np.array([1, 0.5]) #svi brojevi na videu se krecu u desno i na dole
            else:
                aprox_speed = (self.last_location - self.first_location)/number         
            return self.last_location + aprox_speed * self.lost_for


    def position_distance(self, contour):
        """Pronalazi udaljenost između dve pojave na frejmovima. Koristi euklidsku udaljenost centara kontura."""
        # pronađi centar konture
        M = cv2.moments(contour)
        position = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        #uzmi poziciju     
        aprox_position = self.aprox_position()
        #izračunaj euklidsku udaljenost
        dist = math.sqrt( (position[0] - aprox_position [0])**2 + (position[1] - aprox_position[1])**2 )
        return dist

    def distance(self, contour):
        """Pronalazi udaljenost između dve pojave na frejmovima."""
        # ideje: uzeti velicinu o obzir, uzeti broj koji je neuronska mreža pronašla u obzir        
        return self.position_distance(contour)

    def predict(self, contour, image):
        """Vrsi detekciju broja sa konture"""
        x,y,w,h = cv2.boundingRect(contour) 
        region = image[y:y+h+1,x:x+w+1]
        region = iu.process_region(region)
        prediction = self.network.predict([region])        
        self.votes = self.votes + prediction  

    def value(self): 
        return max(enumerate(self.votes[0]), key=lambda x: x[1])[0]

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

    def process_frame(self, contours, frame_number, image):
         """Procesira uočene konture brojeva na jednom frejmu."""
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
                    #pronađen je postokeci
                    min_dist_from_objects['object'].found(contour, frame_number,image)
                    found_in_this_frame[min_dist_from_objects['object'].id] = min_dist_from_objects['object']
                    #izbaci objekat iz object mape, jer ne moze jos neko da bude on
                    del self.objects[min_dist_from_objects['object'].id]
                    continue

             #pokušaj da ga nađeš među izgubljenim
             if len(self.lost_objects):
                min_dist_from_lost = self.find_min_distance(contour, self.lost_objects)
                if min_dist_from_lost['distance'] < 20:
                    min_dist_from_lost['object'].found(contour, frame_number, image)
                    found_in_this_frame[min_dist_from_lost['object'].id] = min_dist_from_lost['object']
                    #izbaci objekat iz object mape, jer ne moze jos neko da bude on
                    del self.lost_objects[min_dist_from_lost['object'].id]
                    continue

             #ovde su nove konture
             obj = TrackingObject(self.next_id, contour, frame_number, image)
             found_in_this_frame[self.next_id] = obj
             self.next_id += 1


         #popravi liste
         #svi koji su ostali u objects su zapravo lost
         self.lost_objects.update(self.objects)
         #javi izgubljenim da su bili jos jedan frejm izgubljeni
         for key,value in self.lost_objects.items():
             value.lost()
         #u objects listu se dodaju oni koji su bili na frejmu
         self.objects = found_in_this_frame

         #proveri obe liste u potrazi za objektima koji su napustili video
         self.check_for_out_of_frame_objects(self.objects)
         self.check_for_out_of_frame_objects(self.lost_objects)


    def find_min_distance(self, contour, objects):
         """Prolazi objekat u listi najbliži konturi."""
         min_obj = next(iter(objects.values()))
         min_dist = min_obj.distance(contour)
         for key, value in objects.items():
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
         for id in ids:
            del objects[id]
    
    def draw_all_traces(self, image):
         self.draw_traces_of_objects(image, self.objects)
         self.draw_traces_of_objects(image, self.lost_objects)
         #self.draw_traces_of_objects(image, self.out_of_frame_objects)

    def draw_traces_of_objects(self, image, objects):
        for key, value in objects.items():
             start = value.first_location
             end = value.aprox_position()
             cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (0,0,255), 2, cv2.LINE_AA) 


    def line_sum(self, line):
        A = self.sum_per_list(line, self.objects)
        B = self.sum_per_list(line, self.lost_objects)
        C = self.sum_per_list(line, self.out_of_frame_objects)
        return  A + B + C

    def sum_per_list(self, line, objects):
        A = Pos(line[0], line[1])
        B = Pos(line[2], line[3])
        sum = 0
        for key, value in objects.items():
            pos = value.aprox_position()
            C = Pos(value.first_location[0], value.first_location[1])
            D = Pos(pos[0], pos[1])
            if self.intersect(A,B,C,D):
                #print(value.value())
                sum += value.value()

        return sum


    def ccw(self,A,B,C):
        return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

    def intersect(self,A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    