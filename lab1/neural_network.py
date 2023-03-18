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
        self.weights_input_hidden_1 = np.random.randn(self.input_size, self.hidden_size_1)
        self.bias_input_hidden_1 = np.random.randn(self.hidden_size_1)
        self.weights_hidden_1_hidden_2 = np.random.randn(self.hidden_size_1, self.hidden_size_2)
        self.bias_hidden_1_hidden_2 = np.random.randn(self.hidden_size_2)
        self.weights_hidden_2_output = np.random.randn(self.hidden_size_2, self.output_size)
        self.bias_hidden_2_output = np.random.randn(self.output_size)
    
    def sigmoid(self, x):   # 使NN分布變成0~1
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_sigmoid(self, x):   # derivative_sigmoid為一個常數，因為x在sigmoid就已經決定了，backward只是後續的步驟
        return np.multiply(x, 1.0 - x)   # 有推導
    
    def relu(self, x):
        return np.maximum(0, x)

    def derivative_relu(self, x):
        z = np.copy(x)
        z[z>0]=1
        z[z<=0]=0
        return z
    
    def forward(self, x, mode):
        if mode == "sigmoid":
            # 輸入層到隱藏層1
            hidden_layer_input_1 = np.dot(x, self.weights_input_hidden_1) + self.bias_input_hidden_1   # (n, 2)* (2, 4) + (4, ) = (n, 4)  ((廣播運算(4, ) = (1, 4)))
            self.hidden_layer_output_1 = self.sigmoid(hidden_layer_input_1)
            # 隱藏層1到隱藏層2
            hidden_layer_input_2 = np.dot(self.hidden_layer_output_1, self.weights_hidden_1_hidden_2) + self.bias_hidden_1_hidden_2  # (n, 4)* (4, 4) + (4, ) = (n, 4)
            self.hidden_layer_output_2 = self.sigmoid(hidden_layer_input_2)
            # 隱藏層2到輸出層
            output_layer_input = np.dot(self.hidden_layer_output_2, self.weights_hidden_2_output) + self.bias_hidden_2_output # (n, 4)* (4, 1) + (1, ) = (n, 1)
            output_layer_output = self.sigmoid(output_layer_input)
            
            return output_layer_output

        elif mode == "relu":
            # 輸入層到隱藏層1
            hidden_layer_input_1 = np.dot(x, self.weights_input_hidden_1) + self.bias_input_hidden_1
            self.hidden_layer_output_1 = self.relu(hidden_layer_input_1)
            # 隱藏層1到隱藏層2
            hidden_layer_input_2 = np.dot(self.hidden_layer_output_1, self.weights_hidden_1_hidden_2) + self.bias_hidden_1_hidden_2
            self.hidden_layer_output_2 = self.relu(hidden_layer_input_2)
            # 隱藏層2到輸出層
            output_layer_input = np.dot(self.hidden_layer_output_2, self.weights_hidden_2_output) + self.bias_hidden_2_output
            output_layer_output = self.relu(output_layer_input)

            return output_layer_output

    def mse_loss(self, y_true, y_pred):  
        #MSE
        return np.mean((y_pred - y_true)**2)
    
    def backward(self, x, y_true, y_pred, mode):  # output_layer_output = y_pred    

        if mode == "sigmoid":
            # 計算輸出層的梯度
            output_layer_error = y_pred - y_true  # (n, 1)
            # output_layer_error = self.mse_loss(y_true, y_pred) # why 用這種loss不行
            output_layer_delta = output_layer_error * self.derivative_sigmoid(y_pred)  # (n, 1)

            # 計算第二個隱藏層的梯度
            hidden_layer_2_error = np.dot(output_layer_delta, self.weights_hidden_2_output.T)  # (n, 1)* (4, 1).T = (n, 4)
            hidden_layer_2_delta = hidden_layer_2_error * self.derivative_sigmoid(self.hidden_layer_output_2)
            
            # 計算第一個隱藏層的梯度
            hidden_layer_1_error = np.dot(hidden_layer_2_delta, self.weights_hidden_1_hidden_2.T)   # (n, 4)* (4, 4).t = (n, 4)
            hidden_layer_1_delta = hidden_layer_1_error * self.derivative_sigmoid(self.hidden_layer_output_1)
        
        elif mode == "relu":
            # 計算輸出層的梯度
            output_layer_error = y_pred - y_true
            output_layer_delta = output_layer_error * self.derivative_relu(y_pred)

            # 計算第二個隱藏層的梯度
            hidden_layer_2_error = np.dot(output_layer_delta, self.weights_hidden_2_output.T)
            hidden_layer_2_delta = hidden_layer_2_error * self.derivative_relu(self.hidden_layer_output_2)
            
            # 計算第一個隱藏層的梯度
            hidden_layer_1_error = np.dot(hidden_layer_2_delta, self.weights_hidden_1_hidden_2.T)
            hidden_layer_1_delta = hidden_layer_1_error * self.derivative_relu(self.hidden_layer_output_1)
                   
        
        # 更新權重和偏置  求delta_C/delta_W = delta_Z/delta_W(可以輕易在forward求出)* delta_C/delta_Z(backward在求此)
        self.weights_hidden_2_output -= self.learning_rate * np.dot(self.hidden_layer_output_2.T, output_layer_delta) # (n, 4).T* (n, 1) = (4, 1)
        self.bias_hidden_2_output -= self.learning_rate * np.sum(output_layer_delta, axis=0)
        self.weights_hidden_1_hidden_2 -= self.learning_rate * np.dot(self.hidden_layer_output_1.T, hidden_layer_2_delta) # (n , 4).T *(n, 4) = (4, 4)
        self.bias_hidden_1_hidden_2 -= self.learning_rate * np.sum(hidden_layer_2_delta, axis=0)
        self.weights_input_hidden_1 -= self.learning_rate * np.dot(x.T, hidden_layer_1_delta) # (n, 2).T* (n, 4) = (2, 4)
        self.bias_input_hidden_1 -= self.learning_rate * np.sum(hidden_layer_1_delta, axis=0)


    def accuracy(self, y_pred, y_true):
        """
        計算模型的精度
        """
        # 四捨五入預測輸出為 0 或 1
        y_pred_round = np.round(y_pred)   # 当整数部分以0结束时，round函数一律是向下取整
        
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
            if i % 1000 == 0:
                acc = self.accuracy(y_pred, y_true)
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
    y_pred = NN.train(x, y, epochs = 20000,  mode = "sigmoid")
    NN.draw_loss_plot()
    NN.show_result(x, y, y_pred)


    # XOR input
    x, y = generate_XOR_easy() 
    NN = NeuralNetwork(2, 4, 4, 1, 0.01)  # init the class
    y_pred = NN.train(x, y, epochs = 100000, mode = "sigmoid")
    NN.draw_loss_plot()
    NN.show_result(x, y, y_pred)
