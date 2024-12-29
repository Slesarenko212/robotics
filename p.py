import GUI
import HAL
import cv2
import numpy as np

p = 0.005 #пропорциональный коэффициент ПИД-регулятора

while True:
    image = HAL.getImage() #получение изображения с камеры
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #преобразование цветового пространства из BGR в HSV
    #нижняя и верхняя границы для маскирования цвета
    lower_thresh = np.array([8, 8, 8])
    upper_thresh = np.array([1, 1, 360])

    console.print("Running")
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh) #маска, выделяющая заданный диапазон цветов
    mask = cv2.bitwise_not(mask) #инвертация маски

    h, w, d = image.shape #размеры изображения

    #верхняя и нижняя границы области поиска
    search_top = 3*h/4
    search_bot = search_top + 20

    M = cv2.moments(mask) #моменты маски для нахождения центра
    if M['m00'] != 0: #если линия найдена
        #координаты центра линии
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1) #круг в центре линии на изображении 

        err = cx - w/2 #ошибка (расстояние центра линии от центра изображения)
        console.print(err)

        GUI.showImage(image) #отображение изображения с нарисованным кругом в центре линии

        HAL.motors.sendV(4) #скорость движения 
        HAL.motors.sendW(p*float(err)) #угловая скорость
