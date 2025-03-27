import cv2
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
ret, frame = cap.read()

def auto_white_balance_histogram(img, percent=1):
    # Разделяем каналы
    b, g, r = cv2.split(img)

    # Функция для коррекции одного канала
    def _balance_channel(channel, percent):
        # Находим минимальный и максимальный пиксели
        min_val = np.percentile(channel, percent)
        max_val = np.percentile(channel, 100 - percent)

        # Линейная коррекция
        channel = np.clip(channel, min_val, max_val)
        channel = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return channel

    # Балансируем каждый канал
    balanced_b = _balance_channel(b, percent)
    balanced_g = _balance_channel(g, percent)
    balanced_r = _balance_channel(r, percent)

    # Собираем обратно
    balanced = cv2.merge([balanced_b, balanced_g, balanced_r])
    return balanced

balanced_frame = auto_white_balance_histogram(frame, percent=1)
blurred = cv2.GaussianBlur(balanced_frame, (0, 0), 3)  #
sharpened = cv2.addWeighted(balanced_frame, 1.5, blurred, -0.5, 0)
cv2.imshow("Auto WB", balanced_frame)
cv2.waitKey(0)
