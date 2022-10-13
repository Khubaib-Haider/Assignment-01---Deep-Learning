import numpy as np
import cv2
import os.path
from knn import KNN

Training_Data = 'F:\Masters RIME\Third Semester\Deep learning\Assignemnts\Test_Images'
Test_Data = 'F:\Masters RIME\Third Semester\Deep learning\Assignments\Train_Images'
Output = ["Balls", "Football"]

def pre_process(path):
    img = cv2.imread(path)
    rsz = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(rsz, cv2.COLOR_BGR2GRAY)
    Test_image = np.asarray(gray_img)
    Test_image = Test_image / 255

    return Test_image

def Extract(folder):
    train_img = []
    for i in os.listdir(folder):
        if (os.path.isfile(folder + "/" + i)):
            Test_image = pre_process(folder + "/" + i)
            train_img.append(Test_image)
    Ans = np.array(train_img)
    return Ans


Ans = []
Ans = Extract(Training_Data)

y_0 = np.zeros(15)
y_1 = np.ones(15)
y = []
y = np.concatenate((y_0, y_1), axis=0)

from builtins import range

num = Ans.shape[0]
mask = list(range(num))
X_train = Ans[mask]
y_train = y[mask]

num_t = Ans.shape[0]
mask = list(range(num_t))
X_test = Ans[mask]
y_test = y[mask]

print("X_train: " + str(X_train.shape))
print("X_test: " + str(X_test.shape))
print("y_train: " + str(y_train.shape))
print("y_test: " + str(y_test.shape))

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

print("X_train: " + str(X_train.shape))
print("X_test: " + str(X_test.shape))
print("y_train: " + str(y_train.shape))
print("y_test: " + str(y_test.shape))

def Test_Model(img):
    Test = pre_process(Test_Data + '/' + img)
    Test = np.reshape(Test, (1, Test.shape[0] * Test.shape[1]))
    classify = KNN()
    classify.train(X_train, y_train)
    distance_L2 = classify.compute_distances(Test)
    y_test_pred = classify.predict_labels(distance_L2, k=1)
    print('Input ' + img + ' as a ' + Output[int(y_test_pred)])


print("Predict the Input Images as")
for n in os.listdir(Test_Data):
    Test_Model(n)