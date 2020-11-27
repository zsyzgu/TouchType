# 可运行的程序

(1) record.py

采集数据的程序。用户分四个session完成“任务.doc”中的前四个任务。record.py用于记录实验过程中的压力板数据（data/user-name/task-id.gz）和录屏（data/user-name/task-id.avi）。

[usage] python record.py user-name/task-id

1. 实验结束时按Esc键保存数据，退出程序

(2) replay.py

人工标注的程序。采集数据完成以后，用户通过replay.py分析录屏和压力板数据，标记出压力板数据中的正例（点击事件）和负例（误触），标注后的数据记录在（data/user-name/task-id_labeled.gz）中。

[usage] python replay.py user-name/task-id

1. 实验结束时按Esc键保存数据，退出程序

2. 鼠标左键将报点标记为正例（绿色），右键将报点标记为负例（红色，所有报点默认就是红色），鼠标中键去除数据（蓝色，用于人分不清该数据是正例还是负例的情况）。

(3) check.py

人工标注以后，我们可以训练出一个初级的二分类模型。运行check.py，然后按空格，程序会用分类器去预测实验数据，如果和用户人工标注的结果不同，那么这个数据点是可疑的，程序会停下来。

这时有两种可能，一是用户当时标注错了，我们可以通过鼠标左右键重新标注（重新标注的结果保存在data/user-name/task-id_checked.gz中）；二是机器学习错了，这时我们便发现了一个机器学习没有正确处理的数据点，应该把它记录下来。

[usage] python check.py user-name/task-id

1. 实验结束时按Esc键保存数据，退出程序

2. 鼠标左键右键重新标注数据。

(4) train.py

根据data/user-name/task-id_checked.gz的实验数据和标注结果，训练二分类模型，并保存在model.pickle中。

[usage] python replay.py user-name/task-id （若user-name为xxx，则表示将所有用户的数据纳入训练；若task-id为x，则表示将所有session的任务纳入训练）

(5) demo.py

展示程序，根据model.pickle中的二分类模型来给正例提供反馈（声音）。

[usage] python demo.py

(6) train2.py & main.py

先不用管，这是古裔正正在开发的深度学习模型。

# 需要说明的类

(1) ContactData类(frame_data.py)

包含一次点击报点的数据，如点击的面积、压力、坐标、椭圆长短轴等等。

(2) FrameData类(frame_data.py)

包含整个压力板的图像数据（force_array）、时间戳（timestamp）和这一帧的点击事件（contacts）。

(3) Board类(board.py)

读取压力板数据，实例化以后自动新开线程来读取并记录压力板数据。

def getFrame(self): 返回最新的一帧（数据类型：FrameData）

def getNewFrame(self): 等待最新的一帧更新，并返回最新的一帧

def getFrameTime(self): 返回最新一帧的时长

