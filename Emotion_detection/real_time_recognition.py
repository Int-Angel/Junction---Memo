'''}
Title: real_time_recognition
Author: COOL TEAM
Date of creation: 10-nov-2023
Last modifiedd: 11-nov-2023
'''
# Load modules
## Load data manipulation libraries
import numpy as np
import pandas as pd
## Load camera manipulation library
import cv2
## Load tensorflow modules for the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

# Create the model architecture and load the model weights
model = Sequential()
## Convolutional (ReLu and MaxPooling) layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
## Output (softmax) layer
model.add(Dense(7, activation='softmax'))
## Load pretrained model weights
model.load_weights('emotion_dataset/face_emotion_classifier.h5') 

# Prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Create a list to store the labels and scores
labels_scores = []

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('Emotion-detection/src/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    #frame = cv2.flip(frame, 1)
    for (x, y, w, h) in faces:
        # Extract face ROI
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        label = emotion_dict[maxindex]
        score = np.max(prediction)
        
        # Store the label and score
        labels_scores.append((label, score))
        
        # Define bounding box and text colors
        colors = {'Angry': (0, 0, 255), 'Disgusted': (0, 0, 255), 'Fearful': (0, 0, 255),
                  'Happy': (0, 255, 0), 'Sad': (0, 0, 255), 'Surprised': (0, 255, 0), 'Neutral': (255,0,0)}
        color = colors[label]

        # Draw bounding box and text on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (255, 255, 255), -1)
        cv2.putText(frame, f'{label}: {score:.2f}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', cv2.resize(frame,(1*800,1*480),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite('test_im.jpg', frame)
scores_df = pd.DataFrame(labels_scores)
scores_df.rename(columns={0:'Labels',1:'Scores'},inplace=True)
num_obs = scores_df.shape[0]
labels_df = scores_df.groupby(by='Labels')

label_counts = labels_df.value_counts()
label_ratio  = label_counts/num_obs
scores_group = np.array(label_ratio.index.get_level_values(1))
label_score_prod = (label_ratio*scores_group).groupby(level=0).sum().sort_values(ascending=False)
print('---------------------------------------------------------------------')
print(label_score_prod)
print('---------------------------------------------------------------------')
print(f'The dominant emotion detected is: {label_score_prod.index[0]} with a {label_score_prod[0]*100:.2f}% among all detections.')
print('---------------------------------------------------------------------')

cap.release()
cv2.destroyAllWindows()
