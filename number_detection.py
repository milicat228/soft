import scipy
import numpy as np
import cv2
import image_utils as iu
import tracker as t

def detect_line(image, channel):
    """Funkcija namenjena za detekciju početne i krajnje tačke linije."""
    #obradi sliku
    img = iu.process_image(image, channel, 0)
    #izdvoj ivice
    img = cv2.Canny(img, 50, 200, None, 3)
    #pronađi konture
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, 180, 10) 
    #Hough transformacija pronađe nekada više linija (po konturama), pa se uzima srednja vrednost   
    x0 = int(scipy.mean(lines[:,0,0]))
    y0 = int(scipy.mean(lines[:,0,1]))
    x1 = int(scipy.mean(lines[:,0,2]))
    y1 = int(scipy.mean(lines[:,0,3]))
    return [x0,y0, x1, y1]
  
        
def process_video(file):
    #učitaj video
    vidcap = cv2.VideoCapture(file)

    #učitaj prvu sliku
    success,image = vidcap.read()
    #izvrši detekciju linija
    blue = detect_line(image, 0)
    green = detect_line(image, 1)
    #nacrtaj linije na slici i sačuvaj sliku
    #copy = image.copy()
    #cv2.line(copy, (blue[0], blue[1]), (blue[2], blue[3]), (0,0,255), 2, cv2.LINE_AA)   
    #cv2.line(copy, (green[0], green[1]), (green[2], green[3]), (0,0,255), 2, cv2.LINE_AA)
    #cv2.imwrite("lines.jpg", copy)

    #kreiranje trackera
    tracker = t.Tracker(blue, green)

    #dalja obrada
    frame_count = 0
    #region_count = 0
    while success:       
        #obradi prethodnu sliku    
        processed_image = iu.process_image(image)
        #cv2.imwrite("images/proccessed_image" + str(frame_count) + ".jpg", processed_image)
        #traženje kontura
        processed_image, contours, hierarchy = cv2.findContours(processed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #izdavajanje brojeva
        detected_contours = []
        for i in range(0, len(contours)):
            contour = contours[i]
            x,y,w,h = cv2.boundingRect(contour)
            #odbacuju se sve konture koje imaju roditelja (npr kontura unuar 0)
            #odbaci premale konture (one su šum)  
            #prvi uslov hvata normalne brojeve, drugi hvata iskošene brojeve 
            if (h >= 15 and h <= 25) or (w > 10 and h >= 14) and (hierarchy[0][i][3] == -1): 
                detected_contours.append(contour)
                #region = processed_image[y:y+h+1,x:x+w+1] 
                cv2.rectangle(image,(x-3,y-3),(x+w+3,y+h+3),(0,255,0),2)
                #region = iu.process_region(region)   
                #prediction = network.predict([region])  
                #p = winner(prediction[0])           
                #cv2.imwrite("area_region_" + str(region_count)+ "_" + str(frame_count)+ "_" + str(p) + ".jpg", region)             
                #region_count += 1
        #cv2.imwrite("images/image_" + str(frame_count)+ ".jpg", processed_image)
        #učitaj sledeću sliku 
        tracker.process_frame(detected_contours, frame_count, processed_image)
        #tracker.draw_all_traces(image)
        #cv2.imwrite("images/tracked" + str(frame_count) + ".jpg", image)
        success,image = vidcap.read()
        frame_count += 1

    #print('blue')
    blue_sum = tracker.line_sum(0)
    #print('green')
    green_sum = tracker.line_sum(1)
    #print('RESULT: ' + str(- green_sum + blue_sum))
    return blue_sum - green_sum


def main():
    res = [-25, -18, -3, -64, -35, 17, -68, 10, -5, 25] 
    student_results = []
    for i in range(0,10):
        print('Working on: ' + 'data/videos/video-' + str(i) + '.avi')
        result = process_video('data/videos/video-' + str(i) + '.avi')
        student_results.append(result)  
        print(str(i + 1) + ' from 10 finished. Result: ' + str(student_results[i]) + ' Expected: ' + str(res[i]))
    diff = 0
    for index, res_col in enumerate(res):
        diff += abs(res_col - student_results[index])
    percentage = 100 - abs(diff/sum(res))*100
    print(student_results)        
    print('Percentage: ' + str(percentage))
            
       
if __name__== "__main__":
  main()
