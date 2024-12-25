import torch
import torch.nn as nn
import torch.optim as optim
import GUI
import HAL
import cv2
import numpy as np

class PIDNN(nn.Module):
    def __init__(self):
        super(PIDNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  
        self.fc2 = nn.Linear(10, 1)   

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = PIDNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

p = 0.0
prev_error = 0
reward = 0.0

while True:
    image = HAL.getImage()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_thresh = np.array([8, 8, 8])
    upper_thresh = np.array([1, 1, 360])

    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    mask = cv2.bitwise_not(mask)

    h, w, d = image.shape
    search_top = 3 * h / 4
    search_bot = search_top + 20

    M = cv2.moments(mask)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)

        err = cx - w / 2
        console.print(err)

        # Преобразование ошибки в тензор
        error_tensor = torch.FloatTensor([[float(err)]])

        # Прогнозирование p с помощью нейросети
        p_pred = model(error_tensor).item()

      
        HAL.motors.sendV(4)
        HAL.motors.sendW(p_pred*err)

        
        reward = -abs(err)  # Чем меньше ошибка, тем больше вознаграждение
        optimizer.zero_grad()
        predicted_p = model(error_tensor)     
        target_tensor = torch.FloatTensor([[reward]])
        loss = loss_fn(predicted_p, target_tensor) 
        loss.backward()
        optimizer.step()

    GUI.showImage(image)
