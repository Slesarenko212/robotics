import torch
import torch.nn as nn
import torch.optim as optim
import GUI
import HAL
import cv2
import numpy as np

#определение класса нейронной сети PIDNN, наследуемого от nn.Module
class PIDNN(nn.Module):
    def __init__(self): #конструктор класса
        super(PIDNN, self).__init__() #инициализация родительского класса
        self.fc1 = nn.Linear(1, 10) #первый полносвязный слой (1 вход, 10 выходов)
        self.fc2 = nn.Linear(10, 1) #второй полносвязный слой (10 входов, 1 выход) 

    def forward(self, x): #метод прямого распространения ошибки
        x = torch.relu(self.fc1(x)) #применение ReLU активации к первому слою
        x = self.fc2(x) #второй слой
        return x #выход

model = PIDNN() #экземпляр нейронной сети
optimizer = optim.Adam(model.parameters(), lr=0.01) #оптимизатор (Adam) для обновления весов модели
loss_fn = nn.MSELoss() #функция потерь (MSE – среднеквадратичная ошибка)

p = 0.0 #начальное значения пропорционального коэффициента ПИД-регулятора
prev_error = 0 #предыдущая ошибка
reward = 0.0 #вознаграждение

while True:
    image = HAL.getImage() #получение изображения с камеры
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #преобразование цветового пространства из BGR в HSV

    #нижняя и верхняя границы для маскирования цвета
    lower_thresh = np.array([8, 8, 8])
    upper_thresh = np.array([1, 1, 360])

    #маска, выделяющая заданный диапазон цветов
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    mask = cv2.bitwise_not(mask) #инвертация маски

    h, w, d = image.shape #размеры изображения 
    
    #верхняя и нижняя границы области поиска 
    search_top = 3 * h / 4
    search_bot = search_top + 20

    M = cv2.moments(mask) #моменты маски для нахождения центра
    if M['m00'] != 0: #если линия найдена
        #координыта центра линии
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1) #круг в центре линии на изображении

        err = cx - w / 2 #ошибка (расстояние центра линии от центра изображения 
        console.print(err)

        #преобразование ошибки в тензор для подачи на вход нейросети
        error_tensor = torch.FloatTensor([[float(err)]])

        #прогнозирование p с помощью нейросети на основе текущей ошибки
        p_pred = model(error_tensor).item()

        HAL.motors.sendV(4) #скорость движения 
        HAL.motors.sendW(p_pred*err) #угловая скорость на основе предсказанного значения p и ошибки

        reward = -abs(err)  #чем меньше ошибка, тем больше вознаграждение
        optimizer.zero_grad() #обнуление градиентов оптимизатора
        predicted_p = model(error_tensor) #предсказанное значение p от модели  
        target_tensor = torch.FloatTensor([[reward]]) #целевой тензор на основе вознаграждения
        
        loss = loss_fn(predicted_p, target_tensor) #функция потерь между предсказанным и целевым значением
        loss.backward() #градиенты потерь 
        optimizer.step() #обновление весов модели 

    GUI.showImage(image) #отображение изображения с нарисованным центром линии
