import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.widgets import CheckButtons

df = pd.read_csv('trial1.csv',names=['Time', 'Zero Order Sensor', '3', 'Gyroscope', 'Tilt Sensor', '6', '7', '8'], header= None)
print(df.head())

def kalman_filter(input_a, input_b, input_c):
    ndata = len(input_a)
    # a11, a12, a21, a22
    a = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    b =  np.zeros((ndata, 2, 2))
    k1 = np.zeros(ndata)
    k2 = np.zeros(ndata)
    theta = np.zeros(ndata)
    bias = np.zeros(ndata)
    theta_min = np.zeros(ndata)
    bias_min = np.zeros(ndata)
    error = np.zeros(ndata)
    RMSE = 0
    signal = input_a - input_b  
    #dtm[i]:=d[i] e[i]; Input Signal Difference of gyro and tilt
    result = np.zeros(ndata)

    # Initial Value of sx11post, sx12post, sx21post, sx22post
    b[ndata-1] = np.array([[0.0,  0.0],
                           [1.0,  0.0]])

    sn = 3.5
    se = 1

    for i in range(ndata):
        # Priori Covar
        c = np.matmul(np.matmul(a, b[i-1]), a.T)

        # Kalman Gain
        # k1[i]:=sx11pri/(sx11pri+sn);
        # k2[i]:=sx21pri/(sx11pri+sn);
        k1[i] = c[0][0] / (c[0][0] + sn)
        k2[i] = c[1][0] / (c[0][0] + sn)

        # Kalman Filtering
        # dthetah[i]:=dthetahmin[i]+k1[i]*(dtm[i]-dthetahmin[i]);
        # dbh[i]:=dbhmin[i]+k2[i]**(dtm[i]-dthetahmin[i]);
        # Where (dtm[i] dthetahmin[i]); is a Innovation Signal
        theta[i] = theta_min[i-1] + k1[i] * (signal[i] - theta_min[i-1])
        bias[i] = bias_min[i-1] + k2[i] * (signal[i] - theta_min[i-1])
        
        # sx11post, sx12post, sx21post, sx22post
        kn = np.array([[1 - k1[i], -k2[i]],
                       [0.0      ,  1.0]])
        b[i] = np.matmul(b[i-1].T, kn).T

        # Prediction
        theta_min[i] = theta[i] + bias[i]
        bias_min[i] = bias[i]

        # Posterioeri Covar Error Update
        result[i] = input_a[i] - theta[i]

        # RMSE
        RMSE = RMSE + ((input_c[i]-result[i])**2) / ndata

        # Error
        error[i] = input_c[i]-result[i]

    return result, k1, k2, error, RMSE

def Correlation(result, zos):
    ndata = len(result)
    Rxy = np.zeros(ndata)
    for i in range (ndata):
        sigmaL = 0
        for j in range (ndata):
            zos_value = zos[j-i]
            if j-i < 0 :
                zos_value = 0
            sigmaL = sigmaL + (result[j]*zos_value)
        Rxy[i] = sigmaL
    return Rxy

fig = plt.figure()
gs1 = gs.GridSpec(8, 4)
ax1 = fig.add_subplot(gs1[:-4, :2])
ax2 = fig.add_subplot(gs1[:-4, 2:])
ax3 = fig.add_subplot(gs1[-4:-2, :2])
ax4 = fig.add_subplot(gs1[-2:, :2])
ax5 = fig.add_subplot(gs1[4:, 2:])

ax1.set( xlabel = 'Time (s)',ylabel = 'Knee Joint Angle (degree)', title= 'Gyroscope')
ax2.set( xlabel = 'Time (s)',ylabel = 'Knee Joint Angle (degree)', title= 'Tilt Sensor')
ax3.set( xlabel = 'Time (s)',ylabel = 'Gain', title= 'K1')
ax4.set( xlabel = 'Time (s)',ylabel = 'Gain', title= 'K2')
ax5.set( xlabel = 'Time (s)',ylabel = 'Amplitudo', title= 'Cross Correlation')

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()

df.plot(ax=ax1, x='Time', y='Gyroscope')
df.plot(ax=ax2, x='Time', y='Tilt Sensor')

time = df['Time'].to_numpy()
gyro = df['Gyroscope'].to_numpy()
tilt = df['Tilt Sensor'].to_numpy()
zero_order = df['Zero Order Sensor'].to_numpy()
result, k1, k2, error, RMSE = kalman_filter(gyro, tilt, zero_order)
Crosscorrel = Correlation(result, zero_order)
ax3.plot(time, k1)
ax4.plot(time, k2)
ax5.plot(time, Crosscorrel)
ax3.legend(['K1'], loc= 0)
ax4.legend(['K2'], loc= 0)
ax5.legend(['Cross'], loc= 0)

fig1 = plt.figure()
fig1.tight_layout(h_pad = 0.5)
gs2 = gs.GridSpec(1, 1)
f1 = np.linspace(0.0, 1.0, 256)
axx1 = fig1.add_subplot(gs2[0, :])

df.plot(ax=axx1, x='Time', y='Zero Order Sensor', legend=False)
df.plot(ax=axx1, x='Time', y='Gyroscope', legend=False)
axx1.plot(time, result)
axx1.plot(time, error)
axx1.legend(['Zero Order Sensor', 'Gyroscope/Unfiltered', 'Kalman Filtering', 'Error'], loc= 0)
axx1.set( xlabel = 'Time (s)',ylabel = 'Knee Joint Angle (degree)', title= 'Kalman Filtering')
axx1.grid()

plot = axx1.get_lines()
lines = ['Zero Order Sensor', 'Gyroscope/Unfiltered', 'Kalman Filtering', 'Error']
loc = plt.axes([0.9, 0.8, 0.1, 0.08])
Legend_button = CheckButtons(loc, lines, [True, True, True, True])

def select_legend(value):
    index = lines.index(value)
    plot[index].set_visible(not plot[index].get_visible())
    plt.draw()
Legend_button.on_clicked(select_legend)

print('RMSE =  '+ str(np.sqrt(RMSE)))

plt.tight_layout()
plt.show()