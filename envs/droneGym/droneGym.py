import gym
import pandas as pd
from gym import spaces
import numpy as np
import json
import os
import time
import csv
from collections import deque

# Physical Constants
m = 0.1         #kg
Ixx = 0.00062   #kg-m^2
Iyy = 0.00113   #kg-m^2
Izz = 0.9*(Ixx + Iyy) #kg-m^2 (Assume nearly flat object, z=0)
dx = 0.114      #m
dy = 0.0825     #m
g = 9.81  #m/s/s
DTR = 1/57.3; RTD = 57.3

thrustCoef = 1.5108 * 10**-5 #kg*m


class droneGym(gym.Env):
    """Custom Environment that follows gym interface"""
    dt: float
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(droneGym, self).__init__()
        self.dt = .01
        self.t = 0
        self.startTime = time.time()

        self.diagPath = r'C:\Users\Stephen\PycharmProjects\QuadcopterSim\visualizer\test\details.csv'
        os.remove(self.diagPath) if os.path.exists(self.diagPath) else None
        with open(self.diagPath,'w',newline = '') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['Time','Simulated Time','Failed on','Reward','Altitude (m)', 'Roll','Pitch','Yaw'])

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Box(low = np.array((0,0,0,0)), high = np.array((100,100,100,100)))
        # self.action_space.n = 4
        self.action_space = spaces.Box(low=np.array((0)), high=np.array((1)))
        self.action_space.n = 1

        # Example for using image as input:
        #current state matrix?
        self.x = self.stateMatrixInit()
        # self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = self.x.shape)
        self.observation_space = self.x

        self.rateLimitUp = 2
        self.rateLimitDown = 8

        #self.observation_space = spaces.Box(low=0, high=255, shape= (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
        self.reward_range = np.array((-np.inf,1))

        self.times = [self.t]
        self.xdot_b = []#latitudinal velocity body frame
        self.ydot_b = []#latitudinal velocity body frame
        self.zdot_b = []#latitudinal velocity body frame
        self.p = []#rotational velocity body frame
        self.q = []#rotational velocity body frame
        self.r = []#rotational velocity body frame
        self.phi = []#euler rotation global frame
        self.theta = []#euler rotation global frame
        self.psi = []#euler rotation global frame
        self.xpos = []#global x position
        self.y = []#global y position
        self.z = []#global z position

        self.u1 = []
        self.u2 = []
        self.u3 = []
        self.u4 = []

        self.prevU = np.zeros(4)

        self.rewardList = []


    def checkActionStepSize(self, action):
        #limit step-to-step action size (imitating motor inertia)
        limitedActions = np.zeros(4)
        for i,n in enumerate(action):
            diff = n - self.prevU[i]
            if diff > self.rateLimitUp:
                limitedActions[i] = self.prevU[i] + self.rateLimitUp
            elif diff < -self.rateLimitDown:
                limitedActions[i] = self.prevU[i] - self.rateLimitDown
            else:
                limitedActions[i] = n

        self.prevU = limitedActions
        return limitedActions

    def step(self, action):
        # Execute one time step within the environment
        if np.isnan(action[0][0]):
            print('nan somehow')
        if len(action[0]<4):
            newAct = np.zeros(4)
            for i,n in enumerate(action[0]):
                newAct[i] = n
            action = newAct

        action[0] = (action[0]/200 + .5) * 100

        action = self.checkActionStepSize(action)

        if np.isnan(action[0]):
            print('tt')

        x_next = self.numericalIntegration(self.x,action,self.dt)
        self.t += self.dt

        if self.t < 1 and x_next[11] < .05:
            x_next[2] = np.min([self.x[2],x_next[2],0])
            x_next[11] = np.max([self.x[11],x_next[11],0])

        reward, done = self.calcReward(self.x)

        x_next[12] = x_next[11] - self.zSet
        x_next[13] = x_next[10] - self.ySet
        x_next[14] = x_next[9] - self.xSet


        self.x = x_next
        self.memory(self.x, action)

        return self.x, reward, done, {}


    def reset(self):
        # Reset the state of the environment to an initial state
        self.t = 0
        self.x = self.stateMatrixInit()
        # self.x[2] = -.049
        # self.x[11] = .049

        self.times = [self.t]
        self.xdot_b = []#latitudinal velocity body frame
        self.ydot_b = []#latitudinal velocity body frame
        self.zdot_b = []#latitudinal velocity body frame
        self.p = []#rotational velocity body frame
        self.q = []#rotational velocity body frame
        self.r = []#rotational velocity body frame
        self.phi = []#euler rotation global frame
        self.theta = []#euler rotation global frame
        self.psi = []#euler rotation global frame
        self.xpos = []#global x position
        self.y = []#global y position
        self.z = []#global z position

        self.u1 = []
        self.u2 = []
        self.u3 = []
        self.u4 = []

        self.prevU = np.zeros(4)

        self.rewardList = []

        return self.x

    def render(self, mode='human', close=False, epNum = 0):
        # Render the environment to the screen
        newfileName = "test" + str(epNum)

        df = pd.DataFrame(list(zip(self.times, self.xdot_b, self.ydot_b, self.zdot_b, self.p, self.q, self.r, self.phi, self.theta, self.psi, self.xpos, self.y, self.z)),
                          columns=['t', 'xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y',
                                   'z'])

        dfAction = pd.DataFrame(list(zip(self.u1, self.u2, self.u3, self.u4)), columns = ['U1','U2','U3','U4'])
        with open(r'C:\Users\Stephen\PycharmProjects\QuadcopterSim\visualizer\test\\' + newfileName + '.js', 'w') as outfile:
            outfile.truncate(0)
            outfile.write("var sim_data = [ \n")
            json.dump([i for i in self.times[0:-1]], outfile, indent=4)
            outfile.write(",\n")
            parsed1 = json.loads(
                dfAction[['U1','U2','U3','U4']].T.to_json(orient='values'))
            json.dump(parsed1, outfile, indent=4)
            outfile.write(",\n")
            parsed = json.loads(
                df[['xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']].T.to_json(
                    orient='values'))
            json.dump(parsed, outfile, indent=4)
            outfile.write("]")



    def calcReward(self, state):
        #calculate reward based off of distance from setpoints
        done = False

        if self.t < 5:
            xSet = 0
            ySet = 0
            zSet = 10
        elif self.t >= 5 and self.t<10:
            xSet = 5
            ySet = 5
            zSet = 10
        elif self.t >= 10 and self.t < 15:
            xSet = -5
            ySet = 5
            zSet = 12
        else:
            xSet = 0
            ySet = 0
            zSet = 2

        self.xSet = xSet
        self.ySet = ySet
        self.zSet = zSet

        reward_unlog = 1 - np.sqrt(self.distin3d(state[9], state[10], state[11], xSet, ySet, zSet))
        reward = 10 * np.exp(reward_unlog)/(np.exp(reward_unlog) + 1)
        reward = round(reward)
        self.rewardList.append(reward)

        maxAngleAllowed = 1#0.5745329
        # pitch_bad = not(-maxAngleAllowed < state[6] < maxAngleAllowed) and self.t > .4
        # roll_bad = not(-maxAngleAllowed < state[7] < maxAngleAllowed) and self.t > .4
        roll_bad = ((2*np.pi)-maxAngleAllowed) > state[6] > maxAngleAllowed and self.t > .4
        pitch_bad = ((2*np.pi)-maxAngleAllowed) > state[7] > maxAngleAllowed and self.t > .4
        alt_bad = not(.05 < state[11] < 100) and self.t > 1

        if self.t > 9.8 or pitch_bad or roll_bad or alt_bad:
            # reward = -200
            if self.t > 9.8:
                reward = 200
                self.rewardList.append(reward)
            else:
                reward = -100
                self.rewardList.append(reward)
            done = True

            if pitch_bad:
                failer = "Pitch"
            elif roll_bad:
                failer = 'Roll'
            else:
                failer = "Alt"

            with open(self.diagPath, 'a', newline = '') as csvFile:
                writer = csv.writer(csvFile)
                totReward = np.sum(self.rewardList)
                writer.writerow([round(time.time()-self.startTime,1), round(self.t,2), failer, totReward, round(state[11],3),round(state[6],3),round(state[7],3),round(state[8],3)])


        return reward, done


    def distin3d(self,x1,y1,z1,x2,y2,z2):
        #calculate distance in 3d space
        return(np.sqrt(np.power(x2-x1,2) + np.power(y2-y1,2) + np.power(z2-z1,2)))

    def memory(self,x, action):
        self.times.append(self.times[-1] + self.dt)
        self.xdot_b.append(x[0])
        self.ydot_b.append(x[1])
        self.zdot_b.append(x[2])
        self.p.append(x[3])
        self.q.append(x[4])
        self.r.append(x[5])
        self.phi.append(x[6])
        self.theta.append(x[7])
        self.psi.append(x[8])
        self.xpos.append(x[9])
        self.y.append(x[10])
        self.z.append(x[11])

        self.u1.append(action[0])
        self.u2.append(action[1])
        self.u3.append(action[2])
        self.u4.append(action[3])

    def stateMatrixInit(self):
        x = np.zeros(15)
        # x[2] = -.049
        x[11] = .049
        x[12] = 9.951
        # x0 = xdot_b = latitudinal velocity body frame
        # x1 = ydot_b = latitudinal velocity body frame
        # x2 = zdot_b = latitudinal velocity body frame
        # x3 = p = rotational velocity body frame
        # x4 = q = rotational velocity body frame
        # x5 = r = rotational velocity body frame
        # x6 = phi = euler rotation global frame
        # x7 = theta = euler rotation global frame
        # x8 = psi = euler rotation global frame
        # x9 = x = global x position
        # x10 = y = global y position
        # x11 = z = global z position
        return x

    def processControlInputs(self, u):

        #linearized motor response
        w_o = np.zeros(4)
        thrustForce = np.zeros(4)

        modifyDef = 150000   #initial Val = 10000000
        lesDef = .013385701848569465 * modifyDef

        for i,n in enumerate(u):
            # thrustForce[i] = .447675* n / 10
            try:
                w_o[i] = modifyDef *(-2/(1+np.e**((n/10)-5)) + 2) - lesDef #rough log equation mapping control signal (voltage) to rps
            except FloatingPointError as e:
                w_o[i] = modifyDef
            thrustForce[i] = thrustCoef * w_o[i]

        F1 = thrustForce[0] + thrustForce[2] + thrustForce[3]/2
        F2 = thrustForce[0] - thrustForce[1] - thrustForce[3]/2
        F3 = thrustForce[0] - thrustForce[2] + thrustForce[3]/2
        F4 = thrustForce[0] + thrustForce[1] - thrustForce[3]/2

        return F1, F2, F3, F4

    def stateTransition(self, x, u):
        xdot = np.zeros(15)

        # Store values in a readable format
        ub = x[0]
        vb = x[1]
        wb = x[2]
        p = x[3]
        q = x[4]
        r = x[5]
        phi = x[6]
        theta = x[7]
        psi = x[8]
        xE = x[9]
        yE = x[10]
        hE = x[11]

        F1, F2, F3, F4 = self.processControlInputs(u)
        # Calculate forces from propeller inputs
        # F1 = u#Fthrust(x, u[0], dx, dy)
        # F2 = u#Fthrust(x, u[1], -dx, -dy)
        # F3 = u#Fthrust(x, u[2], dx, -dy)
        # F4 = u#Fthrust(x, u[3], -dx, dy)
        Fz = F1 + F2 + F3 + F4
        # L = (F1 + F4) * dy - (F2 + F3) * dy
        # M = (F1 + F3) * dx - (F2 + F4) * dx
        # N = 0  # -T(F1, dx, dy) - T(F2, dx, dy) + T(F3, dx, dy) + T(F4, dx, dy)

        L = dy * (F4 - F2)
        M = dx * (F1 - F3)
        N = .01 * (F1 - F2 + F3 - F4)  # .01 = drag coef?  random scaling for yaw

        # Pre-calculate trig values
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # Calculate the derivative of the state matrix using EOM
        xdot[0] = (1/m) * (g*sthe)
        xdot[1] = g * sphi / m
        xdot[2] = (1 / m) * (-Fz) + (g * cphi * cthe)
        xdot[3] = 1 / Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
        xdot[4] = 1 / Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
        xdot[5] = 1 / Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
        xdot[6] = p + (q * sphi + r * cphi) * sthe / cthe  # = phidot
        xdot[7] = q * cphi - r * sphi  # = thetadot
        xdot[8] = (q * sphi + r * cphi) / cthe  # = psidot
        xdot[9] = cthe * cpsi * ub + (-cthe * spsi + sphi * sthe * cpsi) * vb + \
                  (sphi * spsi + cphi * sthe * cpsi) * wb  # = xEdot
        xdot[10] = cthe * spsi * ub + (cphi * cpsi + sphi * sthe * spsi) * vb + \
                   (-sphi * cpsi + cphi * sthe * spsi) * wb  # = yEdot
        xdot[11] = -1 * (-sthe * ub + sphi * cthe * vb + cphi * cthe * wb)  # = zEdot

        #keep the target setpoints the same for now
        xdot[12] = x[12]
        xdot[13] = x[13]
        xdot[14] = x[14]


        return xdot

    def numericalIntegration(self, x, action, dt):
        # for now accept whatever we get from the derivative, maybe in future use Runge
        x_next = x + self.stateTransition(x, action) * dt

        for i,n in enumerate(x_next):
            if i in [0,1,2,9,10,11,12,13,14]:
                continue
            else:
                if np.abs(n)>2*np.pi:
                    x_next[i] = n % (2*np.pi)

        if np.sum(np.isnan(x_next)):
            print('tt')

        return x_next

    def calculateError(self, x, setpoints):
        # store current errors?
        # setpoints = [alt, roll, pitch, yaw]
        altError = setpoints[0] - x[11]
        rollError = setpoints[1] - x[3]
        pitchError = setpoints[2] - x[4]
        yawError = setpoints[3] - x[5]

        return altError, rollError, pitchError, yawError

