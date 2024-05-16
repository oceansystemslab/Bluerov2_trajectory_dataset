
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import pandas as pd
import matplotlib.pyplot as plt
torch.manual_seed(50)
LEN_STATE = 6
SAVE = True
LOAD = False
LOAD_PATH = 'linux_inferiority_model.pt'
hidden1 = 256

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(12, hidden1)
        self.act1 = nn.Tanh()
        self.hidden2 = nn.Linear(hidden1, hidden1)
        self.act2 = nn.Tanh()
        self.output = nn.Linear(hidden1, 6)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.output(x)
        return x
 
model = NNModel()
model.to(torch.float64)



df = pd.read_csv('odom-12-01-2024-run1.csv', sep=',', header=0)
#zvalues_numpy = df.values.astype(np.float64)
torch_data = torch.tensor(df.values)

#torch_data = torch_data.to(torch.double)
# V0 
v_t0 = torch_data[:-1, 8:14]
# 
F_t0 = torch_data[:-1, 20:26]
# inputs
X = torch.cat((v_t0,F_t0),1)
print(torch.min(v_t0), torch.max(v_t0))
print(torch.min(F_t0), torch.max(F_t0))

# sx
v_t1 = torch_data[1:, 8:14]
y = v_t1 - v_t0
print(torch.min(y), torch.max(y))



print(v_t0.shape, v_t1.shape)
#print(F_t0[1:40])
print(X.shape, X.dtype)
print(X)
#exit()
if not LOAD:

    loss_fn = nn.MSELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100
    batch_size = 100
     
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):

            Xbatch = X[i:i+batch_size]
            # exit()
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')


else:
    
    model.load_state_dict(torch.load(LOAD_PATH))
#
StartPoint = 500
LEN = 200

v0 = v_t0[StartPoint]
print(v0)



full_state = np.zeros((LEN+1,LEN_STATE))
full_state[0] = v0
for i in range(StartPoint,StartPoint+LEN):
    
    dv  = model(torch.cat((v0,F_t0[i]),0))
    # print(i, v0, dv, v0+dv)
    #input()
    v0 = v0 # + dv 
    full_state[i-StartPoint+1] = v0.detach().numpy()




if SAVE:
    torch.save(model.state_dict(), "mac_model.pt")
    #exit()


dt = 0.1
time = np.linspace(0, LEN/dt, LEN+1) 


v_t0 = v_t0.detach().numpy()
#print(v_t0[0:LEN,0].shape, time[0:LEN].shape, v_t0[StartPoint:StartPoint+LEN,0].shape, )
#exit()

plt.plot(time[0:LEN], v_t0[StartPoint:StartPoint+LEN,0], linewidth=2)
plt.plot(time[0:LEN], full_state[0:LEN,0], linewidth=2)
# plt.legend(['$v_x \, True$', '$v_x \, Koopman$'])
plt.legend(['True Bu', 'NN'])
# plt.grid(axis='both', color='0.95')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [s]')
plt.savefig('vx_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.plot(time[0:LEN], v_t0[StartPoint:StartPoint+LEN,1], linewidth=2)
plt.plot(time[0:LEN], full_state[0:LEN,1], linewidth=2)
# plt.legend(['$v_x \, True$', '$v_x \, Koopman$'])
plt.legend(['True Bw', 'NN'])
# plt.grid(axis='both', color='0.95')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [s]')
plt.savefig('vy_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.plot(time[0:LEN], v_t0[StartPoint:StartPoint+LEN,2], linewidth=2)
plt.plot(time[0:LEN], full_state[0:LEN,2], linewidth=2)
# plt.legend(['$v_x \, True$', '$v_x \, Koopman$'])
plt.legend(['True Bp', 'NN'])
# plt.grid(axis='both', color='0.95')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [s]')
plt.savefig('vz_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()
