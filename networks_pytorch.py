import threading
import abc
import numpy as np

import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Network:
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, 
                shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        
        inp = None
        if hasattr(self, 'num_steps'):
            inp = (self.num_steps, input_dim)
        else:
            inp = (self.input_dim,)

        # 공유 신경망 사용
        self.head = None
        if self.shared_network is None:
            self.head = self.get_network_head(inp, self.output_dim)
        else:
            self.head = self.shared_network
        
        # 공유 신경망 미사용
        # self.head = self.get_network_head(inp, self.output_dim)

        self.model = torch.nn.Sequential(self.head)
        if self.activation == 'linear':
            pass
        elif self.activation == 'relu':
            self.model.add_module('activation', torch.nn.ReLU())
        elif self.activation == 'leaky_relu':
            self.model.add_module('activation', torch.nn.LeakyReLU())
        elif self.activation == 'sigmoid':
            self.model.add_module('activation', torch.nn.Sigmoid())
        elif self.activation == 'tanh':
            self.model.add_module('activation', torch.nn.Tanh())
        elif self.activation == 'softmax':
            self.model.add_module('activation', torch.nn.Softmax(dim=1))
        self.model.apply(Network.init_weights)
        self.model.to(device)

        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
        self.criterion = None
        if loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif loss == 'binary_crossentropy':
            self.criterion = torch.nn.BCELoss()

    # def predict(self, sample):
    #     with self.lock:
    #         self.model.eval()
    #         with torch.no_grad():
    #             x = torch.from_numpy(sample).float().to(device)
    #             pred = self.model(x).detach().cpu().numpy()
    #             pred = pred.flatten()
    #         return pred
        
    def predict(self, sample):
        with self.lock:  # 테스트 중에 스레드 안전성을 보장하기 위해 락을 획득합니다.
            self.model.eval()  # 모델을 평가 모드로 설정하여 드롭아웃 및 배치 정규화 동작을 비활성화합니다.
            with torch.no_grad():  # 예측 시에는 기울기 계산을 끄기 위해 torch.no_grad() 내에서 연산합니다.

                # 입력 데이터를 PyTorch 텐서로 변환하고 해당 디바이스(GPU/CPU)로 이동합니다.
                x = torch.from_numpy(sample).float().to(device)

                # 모델을 통과시켜 예측 결과를 계산하고, 그 결과를 넘파이 배열로 변환합니다.
                # detach()를 사용하여 예측 결과에 연결된 계산 그래프를 끊어서 메모리 사용을 줄입니다.
                # cpu()를 사용하여 예측 결과를 CPU로 이동합니다.
                pred = self.model(x).detach().cpu().numpy()

                # 2D 예측 결과를 1D 배열로 평평화합니다.
                pred = pred.flatten()

            return pred  # 최종 예측 결과를 반환합니다.


    # def train_on_batch(self, x, y):
    #     loss = 0.
    #     with self.lock:
    #         self.model.train()
    #         x = torch.from_numpy(x).float().to(device)
    #         y = torch.from_numpy(y).float().to(device)
    #         y_pred = self.model(x)
    #         _loss = self.criterion(y_pred, y)
    #         self.optimizer.zero_grad()
    #         _loss.backward()
    #         self.optimizer.step()
    #         loss += _loss.item()
    #     return loss
    
    def train_on_batch(self, x, y):
        loss = 0.  # 배치별 누적 손실을 추적하기 위한 변수를 초기화합니다.

        with self.lock:  # 훈련 중에 스레드 안전성을 보장하기 위해 락을 획득합니다.
            self.model.train()  # 모델을 훈련 모드로 설정하여 기울기 계산을 활성화합니다.

            # 입력 데이터(x)와 타겟 레이블(y)을 PyTorch 텐서로 변환하고 해당 디바이스(GPU/CPU)로 이동합니다.
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)

            # 순방향 전파: 입력 데이터를 모델을 통과시켜 예측 결과(y_pred)를 계산합니다.
            y_pred = self.model(x)

            # 손실을 계산하기 위해 예측 결과(y_pred)와 타겟 레이블(y)을 비교합니다.
            _loss = self.criterion(y_pred, y)

            # 역전파: 모델 파라미터에 대한 손실의 기울기를 계산합니다.
            self.optimizer.zero_grad()  # 이전에 저장된 기울기를 옵티마이저에서 초기화합니다.
            _loss.backward()  # 역전파를 사용하여 기울기를 계산합니다.

            # 계산된 기울기로 모델 파라미터를 업데이트합니다 (옵티마이저의 업데이트 단계).
            self.optimizer.step()

            # 이 배치에 대한 손실 값을 누적 손실에 추가합니다.
            loss += _loss.item()

        return loss  # 이 배치에 대한 누적 손실을 반환합니다.


    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
        if net == 'dnn':
            return DNN.get_network_head((input_dim,), output_dim)
        elif net == 'lstm':
            return LSTMNetwork.get_network_head((num_steps, input_dim), output_dim)
        elif net == 'cnn':
            return CNN.get_network_head((num_steps, input_dim), output_dim)

    @abc.abstractmethod
    def get_network_head(inp, output_dim):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init.normal_(weight, std=0.01)

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            torch.save(self.model, model_path)

    def load_model(self, model_path):
        if model_path is not None:
            self.model = torch.load(model_path)
    
class DNN(Network):
    @staticmethod
    def get_network_head(inp, output_dim):
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Linear(inp[0], 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)


class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            LSTMModule(inp[1], 128, batch_first=True, use_last_only=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim))
        return super().predict(sample)


class LSTMModule(torch.nn.LSTM):
    def __init__(self, *args, use_last_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_last_only = use_last_only

    def forward(self, x):
        output, (h_n, _) = super().forward(x)
        if self.use_last_only:
            return h_n[-1]
        return output


class CNN(Network):
    def __init__(self, *args, num_steps=50, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_network_head(inp, output_dim):
        kernel_size = 3
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Conv1d(inp[0], 1, kernel_size),
            torch.nn.BatchNorm1d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(inp[1] - (kernel_size - 1), 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )

    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)


# import torch

# class CNN(Network):
#     def __init__(self, *args, num_steps=1, **kwargs):
#         self.num_steps = num_steps
#         super().__init__(*args, **kwargs)

#     @staticmethod
#     def get_network_head(inp, output_dim):
#         kernel_size = 3  # Conv1d 레이어의 커널 사이즈를 3으로 변경합니다.
#         stride = 1  # Conv1d 레이어에 적용할 스트라이드를 1로 설정합니다.
#         padding = 1  # Conv1d 레이어에 적용할 패딩을 1로 설정합니다. (SAME 패딩)

#         # Sequential 모델로 레이어를 순차적으로 정의합니다.
#         return torch.nn.Sequential(
#             torch.nn.Conv1d(inp[0], 64, kernel_size, stride=stride, padding=padding),  # 64개의 아웃풋 채널을 가진 첫 번째 Conv1d 레이어를 추가합니다.
#             torch.nn.BatchNorm1d(64),  # 첫 번째 Conv1d 레이어의 아웃풋에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Conv1d(64, 64, kernel_size, stride=stride, padding=padding),  # 64개의 아웃풋 채널을 가진 두 번째 Conv1d 레이어를 추가합니다.
#             torch.nn.BatchNorm1d(64),  # 두 번째 Conv1d 레이어의 아웃풋에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Conv1d(64, 32, kernel_size, stride=stride, padding=padding),  # 32개의 아웃풋 채널을 가진 세 번째 Conv1d 레이어를 추가합니다.
#             torch.nn.BatchNorm1d(32),  # 세 번째 Conv1d 레이어의 아웃풋에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Conv1d(32, 16, kernel_size, stride=stride, padding=padding),  # 16개의 아웃풋 채널을 가진 네 번째 Conv1d 레이어를 추가합니다.
#             torch.nn.BatchNorm1d(16),  # 네 번째 Conv1d 레이어의 아웃풋에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Flatten(),  # 출력을 1차원 벡터로 평탄화합니다.
#             torch.nn.Dropout(p=0.1),  # 10%의 확률로 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.
#             torch.nn.Linear(inp[1] * 16, 128),  # 첫 번째 선형 레이어를 정의합니다. 첫 번째 선형 레이어의 입력 크기를 업데이트합니다.
#             torch.nn.BatchNorm1d(128),  # 128차원의 출력에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Dropout(p=0.1),  # 10%의 확률로 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.
#             torch.nn.Linear(128, 64),  # 두 번째 선형 레이어를 정의합니다.
#             torch.nn.BatchNorm1d(64),  # 64차원의 출력에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Dropout(p=0.1),  # 10%의 확률로 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.
#             torch.nn.Linear(64, 32),  # 세 번째 선형 레이어를 정의합니다.
#             torch.nn.BatchNorm1d(32),  # 32차원의 출력에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Dropout(p=0.1),  # 10%의 확률로 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.
#             torch.nn.Linear(32, output_dim),  # 최종 출력 차원과 연결되는 선형 레이어를 정의합니다.
#         )

#     def train_on_batch(self, x, y):
#         x = torch.tensor(x, dtype=torch.float32).reshape((-1, self.num_steps, self.input_dim))
#         return super().train_on_batch(x, y)

#     def predict(self, sample):
#         sample = torch.tensor(sample, dtype=torch.float32).reshape((1, self.num_steps, self.input_dim))
#         return super().predict(sample)



# import torch

# class CNN(Network):
#     def __init__(self, *args, num_steps=1, **kwargs):
#         self.num_steps = num_steps
#         super().__init__(*args, **kwargs)

#     @staticmethod
#     def get_network_head(inp, output_dim):
#         kernel_size = 3  # Conv1d 레이어의 커널 사이즈를 3으로 변경합니다.
#         stride = 1  # Conv1d 레이어에 적용할 스트라이드를 1로 설정합니다.
#         padding = 1  # Conv1d 레이어에 적용할 패딩을 1로 설정합니다. (SAME 패딩)

#         # Sequential 모델로 레이어를 순차적으로 정의합니다.
#         return torch.nn.Sequential(
#             torch.nn.Conv1d(inp[0], 64, kernel_size, stride=stride, padding=padding),  # 64개의 아웃풋 채널을 가진 첫 번째 Conv1d 레이어를 추가합니다.
#             torch.nn.BatchNorm1d(64),  # 첫 번째 Conv1d 레이어의 아웃풋에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Conv1d(64, 64, kernel_size, stride=stride, padding=padding),  # 64개의 아웃풋 채널을 가진 두 번째 Conv1d 레이어를 추가합니다.
#             torch.nn.BatchNorm1d(64),  # 두 번째 Conv1d 레이어의 아웃풋에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Conv1d(64, 32, kernel_size, stride=stride, padding=padding),  # 32개의 아웃풋 채널을 가진 세 번째 Conv1d 레이어를 추가합니다.
#             torch.nn.BatchNorm1d(32),  # 세 번째 Conv1d 레이어의 아웃풋에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Conv1d(32, 16, kernel_size, stride=stride, padding=padding),  # 16개의 아웃풋 채널을 가진 네 번째 Conv1d 레이어를 추가합니다.
#             torch.nn.BatchNorm1d(16),  # 네 번째 Conv1d 레이어의 아웃풋에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Flatten(),  # 출력을 1차원 벡터로 평탄화합니다.
#             torch.nn.Dropout(p=0.1),  # 10%의 확률로 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.
#             torch.nn.Linear(inp[1] * 16, 128),  # 첫 번째 선형 레이어를 정의합니다. 첫 번째 선형 레이어의 입력 크기를 업데이트합니다.
#             torch.nn.BatchNorm1d(128),  # 128차원의 출력에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Dropout(p=0.1),  # 10%의 확률로 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.
#             torch.nn.Linear(128, 64),  # 두 번째 선형 레이어를 정의합니다.
#             torch.nn.BatchNorm1d(64),  # 64차원의 출력에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Dropout(p=0.1),  # 10%의 확률로 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.
#             torch.nn.Linear(64, 32),  # 세 번째 선형 레이어를 정의합니다.
#             torch.nn.BatchNorm1d(32),  # 32차원의 출력에 배치 정규화를 적용합니다.
#             torch.nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
#             torch.nn.Dropout(p=0.1),  # 10%의 확률로 뉴런을 무작위로 비활성화하여 과적합을 방지합니다.
#             torch.nn.Linear(32, output_dim),  # 최종 출력 차원과 연결되는 선형 레이어를 정의합니다.
#         )

#     def train_on_batch(self, x, y):
#         x = torch.tensor(x, dtype=torch.float32).reshape((-1, self.num_steps, self.input_dim))
#         return super().train_on_batch(x, y)

#     def predict(self, sample):
#         sample = torch.tensor(sample, dtype=torch.float32).reshape((1, self.num_steps, self.input_dim))
#         return super().predict(sample)
