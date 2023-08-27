import pyfirmata
import time
import ecg_plot
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


import Adafruit_CharLCD as LCD

# Define LCD pins (use GPIO pin numbers)
lcd_en = 24
lcd_rs = 25
lcd_d4 = 23
time.sleep(5)
lcd_d5 = 17
lcd_d6 = 18
lcd_d7 = 22
lcd_columns = 16  # Number of columns in your LCD
lcd_rows = 2     # Number of rows in your LCD

# Initialize LCD
lcd = LCD.Adafruit_CharLCD(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows)

# Display a message
lcd.message('Hello,\nLCD!')

# Clear the display after a few seconds
lcd.clear()



'''
board = pyfirmata.Arduino('COM5')
analog_pin = board.analog[0]
it = pyfirmata.util.Iterator(board)
it.start()

board.analog[0].mode = pyfirmata.INPUT


def moving_average(a, n=100):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

beats = []
pbar = tqdm(total=1000)
plt.figure(figsize=(16, 5))
while len(beats) < 1000:
    val = board.analog[0].read()
    if val is None:
        continue
    beats.append(val)
    time.sleep(1 / 125)
    pbar.update(1)

beats = np.array(beats)
beats = moving_average(beats, n=100)

classes = {'Normal Condition': 0, 'Fast Atrial Rhythm': 1, 'BIGU': 2, 'Artificial Heart Stimulation': 3, 'Fast Heart Rhythm': 4, 'Irregular Rhythm': 5, 'Slow Heart Rate': 6, 'Atrial Fibrillation': 7, 'Rapid Heart Rate': 8, 'Variability Detected': 9, 'Fast Supraventricular Rhythm': 10, 'Triple-Gemini Pattern': 11}
classes = {i : k for k, i in classes.items()}

plt.plot(beats)
plt.savefig('ecg.png')

im = Image.open('ecg.png')
im = im.resize((150, 150))
img = np.asarray(im)
# img = np.expand_dims(img, 0)
img = np.expand_dims(img[...,:3], 0)

loaded_model = tf.keras.models.load_model('model.h5', compile=False)
predictions = loaded_model.predict(img)
print(classes[round(predictions[0][0])])
#lcd.message = "((classes[round(predictions[0][0])]))"
'''

