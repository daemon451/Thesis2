import simulator
import pandas as pd
import numpy as np
import shutil
import os
import matplotlib.pyplot as plt


pat = r'C:\Users\Stephen\PycharmProjects\QuadcopterSim\visualizer\test\test1000.js'
times = []

def translateJStoPython(pat):
    pat2 = pat[0:-3] + 'temp.csv'
    cols = ['t', 'u1','u2','u3','u4','xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y','z']
    shutil.copyfile(pat,pat2)

    #find the first ], which is where the t array ends, staying the len of each section.  Then remove , and other brackets
    df2 = pd.read_table(pat2, skiprows = 1)
    df2[df2.columns[0]].str.find(']')
    qq = list(map(lambda x: x == '],', df2[df2.columns[0]]))
    lenOfRun = qq.index(True)

    df2 = df2.replace({',':''},regex = True)
    df2 = df2[pd.to_numeric(df2['['], errors = 'coerce').notnull()]


    #reshape into an array
    dfFinal = pd.DataFrame(np.reshape(df2.to_numpy(), (lenOfRun, len(cols)), order='F'), columns=cols).astype('float')

    os.remove(pat2)

    return dfFinal

if __name__ ==  "__main__":
    plt.ion()
    ds = simulator.droneSim()
    df = translateJStoPython(pat)
    dt = .01

    x = ds.stateMatrixInit()

    for i in range(0, len(df)):
        us = np.array([df.loc[i,'u1'], df.loc[i,'u2'],df.loc[i,'u3'],df.loc[i,'u4']])
        times.append(df.loc[i,'t'])

        x_next, currU = ds.numericalIntegration(x, us, dt, errsIsControl = True)
        if x[11] < .05:
            x_next[2] = np.min([x[2], x_next[2], 0])
            x_next[11] = np.max([x[11], x_next[11], .05])
        ds.memory(x_next)
        x = x_next

    df1 = simulator.plotStuff(times, ds.xdot_b, ds.ydot_b, ds.zdot_b, ds.p, ds.q, ds.r, ds.phi, ds.theta, ds.psi, ds.x,
                             ds.y, ds.z, df['u1'], df['u2'], df['u3'], df['u4'], title = ': Simulator')

    df2 = simulator.plotStuff(times, df.xdot_b, df.ydot_b, df.zdot_b, df.p, df.q, df.r, df.phi, df.theta, df.psi, df.x,
                             df.y, df.z, df['u1'], df['u2'], df['u3'], df['u4'], title = ": Gym")

    print('tt')