import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
        pts = np.random.uniform(0, 1 , (n,2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            distance = (pt[0]-pt[1])/1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化權重和偏置  y = Wx + b
        # 生成一個或多個符合標準正態分佈的隨機數，其均值為0，標準差為1
        self.weights_input_hidden_1 = np.random.randn(self.input_size, self.hidden_size_1)
        self.weights_hidden_1_hidden_2 = np.random.randn(self.hidden_size_1, self.hidden_size_2)
        self.weights_hidden_2_output = np.random.randn(self.hidden_size_2, self.output_size)

        self.bias_input_hidden_1 = np.random.randn(self.hidden_size_1)
        self.bias_hidden_1_hidden_2 = np.random.randn(self.hidden_size_2)
        self.bias_hidden_2_output = np.random.randn(self.output_size)

    def sigmoid(self, x):   # 使NN分布變成0~1
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_sigmoid(self, x):   # derivative_sigmoid為一個常數，因為x在sigmoid就已經決定了，backward只是後續的步驟
        return np.multiply(x, 1.0 - x)   # 有推導
    
    # 會有Dead ReLU Problem
    # 產生這種現象的兩個原因：參數初始化問題；learning rate太高導致在訓練過程中參數更新太大。
    # solve: 1.leaky relu 2.Batch Normalization 的歸依化操作
    def relu(self, x):
        return np.maximum(0.01* x, x)

    def derivative_relu(self, x):
        z = np.copy(x)
        z[z>0]=1
        z[z<=0]=0.01
        return z
    

    
    def forward(self, x, mode):
        if mode == "sigmoid":
            # 輸入層到隱藏層1
            hidden_input_1 = np.dot(x, self.weights_input_hidden_1) + self.bias_input_hidden_1   # (n, 2)* (2, 4) + (4, ) = (n, 4)  ((廣播運算(4, ) = (1, 4)))
            self.hidden_output_1 = self.sigmoid(hidden_input_1)
            # 隱藏層1到隱藏層2
            hidden_input_2 = np.dot(self.hidden_output_1, self.weights_hidden_1_hidden_2) + self.bias_hidden_1_hidden_2  # (n, 4)* (4, 4) + (4, ) = (n, 4)
            self.hidden_output_2 = self.sigmoid(hidden_input_2)
            # 隱藏層2到輸出層
            output_input = np.dot(self.hidden_output_2, self.weights_hidden_2_output) + self.bias_hidden_2_output # (n, 4)* (4, 1) + (1, ) = (n, 1)
            output = self.sigmoid(output_input)
            return output

        elif mode == "relu":
            # 輸入層到隱藏層1
            hidden_input_1 = np.dot(x, self.weights_input_hidden_1) + self.bias_input_hidden_1
            self.hidden_output_1 = self.relu(hidden_input_1)
            # 隱藏層1到隱藏層2
            hidden_input_2 = np.dot(self.hidden_output_1, self.weights_hidden_1_hidden_2) + self.bias_hidden_1_hidden_2
            self.hidden_output_2 = self.relu(hidden_input_2)
            # 隱藏層2到輸出層
            output_input = np.dot(self.hidden_output_2, self.weights_hidden_2_output) + self.bias_hidden_2_output
            output = self.relu(output_input)
            return output
        
        elif mode == "None":
            # 輸入層到隱藏層1
            hidden_input_1 = np.dot(x, self.weights_input_hidden_1) + self.bias_input_hidden_1   # (n, 2)* (2, 4) + (4, ) = (n, 4)  ((廣播運算(4, ) = (1, 4)))
            self.hidden_output_1 = hidden_input_1
            # 隱藏層1到隱藏層2
            hidden_input_2 = np.dot(self.hidden_output_1, self.weights_hidden_1_hidden_2) + self.bias_hidden_1_hidden_2  # (n, 4)* (4, 4) + (4, ) = (n, 4)
            self.hidden_output_2 = hidden_input_2
            # 隱藏層2到輸出層
            output_input = np.dot(self.hidden_output_2, self.weights_hidden_2_output) + self.bias_hidden_2_output # (n, 4)* (4, 1) + (1, ) = (n, 1)
            output = output_input
            return output

    def mse_loss(self, y_true, y_pred):  
        return np.mean((y_true - y_pred)**2)
    
    def derivative_mse_loss(self, y_true, y_pred):   # 對y_pred微分
        return -2* (y_true - y_pred)
    
    def backward(self, x, y_true, y_pred, mode):  # output = y_pred    
        if mode == "sigmoid":
            # 計算輸出層的梯度
            output_error = self.derivative_mse_loss(y_true, y_pred)  # (n, 1)
            output_delta = output_error * self.derivative_sigmoid(y_pred)  # (n, 1)

            # 計算第二個隱藏層的梯度
            hidden_2_error = np.dot(output_delta, self.weights_hidden_2_output.T)  # (n, 1)* (4, 1).T = (n, 4)
            hidden_2_delta = hidden_2_error * self.derivative_sigmoid(self.hidden_output_2)
            
            # 計算第一個隱藏層的梯度
            hidden_1_error = np.dot(hidden_2_delta, self.weights_hidden_1_hidden_2.T)   # (n, 4)* (4, 4).t = (n, 4)
            hidden_1_delta = hidden_1_error * self.derivative_sigmoid(self.hidden_output_1)
        
        elif mode == "relu":
            # 計算輸出層的梯度
            output_error = self.derivative_mse_loss(y_true, y_pred)
            output_delta = output_error * self.derivative_relu(y_pred)

            # 計算第二個隱藏層的梯度
            hidden_2_error = np.dot(output_delta, self.weights_hidden_2_output.T)
            hidden_2_delta = hidden_2_error * self.derivative_relu(self.hidden_output_2)
            
            # 計算第一個隱藏層的梯度
            hidden_1_error = np.dot(hidden_2_delta, self.weights_hidden_1_hidden_2.T)
            hidden_1_delta = hidden_1_error * self.derivative_relu(self.hidden_output_1)

        elif mode == "None":
            # 計算輸出層的梯度
            output_error = self.derivative_mse_loss(y_true, y_pred)
            output_delta = output_error 

            # 計算第二個隱藏層的梯度
            hidden_2_error = np.dot(output_delta, self.weights_hidden_2_output.T)
            hidden_2_delta = hidden_2_error
            
            # 計算第一個隱藏層的梯度
            hidden_1_error = np.dot(hidden_2_delta, self.weights_hidden_1_hidden_2.T)
            hidden_1_delta = hidden_1_error
                   
        
        # 更新權重和偏置  求delta_C/delta_W = delta_Z/delta_W(可以輕易在forward求出)* delta_C/delta_Z(backward在求此)
        self.weights_hidden_2_output -= self.learning_rate * np.dot(self.hidden_output_2.T, output_delta) # (n, 4).T* (n, 1) = (4, 1)
        self.bias_hidden_2_output -= self.learning_rate * np.sum(output_delta, axis=0)
        self.weights_hidden_1_hidden_2 -= self.learning_rate * np.dot(self.hidden_output_1.T, hidden_2_delta) # (n , 4).T *(n, 4) = (4, 4)
        self.bias_hidden_1_hidden_2 -= self.learning_rate * np.sum(hidden_2_delta, axis=0)
        self.weights_input_hidden_1 -= self.learning_rate * np.dot(x.T, hidden_1_delta) # (n, 2).T* (n, 4) = (2, 4)
        self.bias_input_hidden_1 -= self.learning_rate * np.sum(hidden_1_delta, axis=0)


    def accuracy(self, y_true, y_pred):
        # 四捨五入預測輸出為 0 或 1
        y_pred_round = []
        for i in range(y_pred.shape[0]):
            if y_pred[i] > 0.5:
                y_pred_round.append(1)
            else:
                y_pred_round.append(0)
        y_pred_round = np.array(y_pred_round).reshape((y_pred.shape[0], 1))

        # 將預測輸出與真實標籤進行比較
        correct = (y_pred_round == y_true).sum()
        total = y_true.shape[0]
        
        # 返回精度
        return correct / total
    
    def train(self, x_train, y_train, epochs, mode):
        self.loss_list = []
        for i in range(epochs):
            y_true = y_train  # 在此ground truth用和train一樣的資料 y為顏色

            # 正向傳播
            y_pred = self.forward(x_train, mode)

            # 計算損失和梯度，反向傳播更新權重和偏置
            loss = self.mse_loss(y_true, y_pred)
            self.loss_list.append(loss)
            self.backward(x, y_true, y_pred, mode)
            
            # 每 1000 個 epoch 打印一次損失
            if i % 2000 == 0:
                acc = self.accuracy(y_true, y_pred)
                print("Epoch {}, Loss: {:.4f}, Acc: {:.4f}".format(i, loss, acc))
        print("End training!!!")

        # 因為train 會重複使用所以再train下一個之前要重新將weight變成random
        self.weights_input_hidden_1 = np.random.randn(self.input_size, self.hidden_size_1)
        self.bias_input_hidden_1 = np.random.randn(self.hidden_size_1)
        self.weights_hidden_1_hidden_2 = np.random.randn(self.hidden_size_1, self.hidden_size_2)
        self.bias_hidden_1_hidden_2 = np.random.randn(self.hidden_size_2)
        self.weights_hidden_2_output = np.random.randn(self.hidden_size_2, self.output_size)
        self.bias_hidden_2_output = np.random.randn(self.output_size)
        return y_pred

    def draw_loss_plot(self):
        fig, ax = plt.subplots()
        ax.plot(np.arange((len(self.loss_list))), self.loss_list)
        # print(len(self.loss_list))
        plt.xlabel('epoch')
        plt.ylabel('Loss(MSE)')
        plt.show()

    def show_result(self, x, y, pred_y):
        plt.subplot(1,2,1)
        plt.title('Ground truth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] < 0.5:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1],'bo')

        plt.subplot(1,2,2)
        plt.title('Predict result' , fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i] < 0.5:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')

        plt.show()

if __name__ == '__main__':
    # random input
    x, y = generate_linear(n = 100)
    NN = NeuralNetwork(2, 4, 4, 1, 0.01)  # init the class
    y_pred = NN.train(x, y, epochs = 10000,  mode = "sigmoid")   # sigmoid要10000配0.01 -> acc = 1.0; relu要2000配0.001; None要200配0.0001 -> acc = 0.99
    NN.draw_loss_plot()
    NN.show_result(x, y, y_pred)

    # XOR input
    x, y = generate_XOR_easy() 
    NN = NeuralNetwork(2, 4, 4, 1, 0.01)  # init the class
    y_pred = NN.train(x, y, epochs = 30000, mode = "sigmoid")  # sigmoid要30000配0.01 -> acc = 1.0; relu要10000配0.001(有時候會失敗); None無法
    NN.draw_loss_plot()
    NN.show_result(x, y, y_pred)
