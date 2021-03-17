import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIDcontrol
import json

np.seterr(all='raise')


# Physical Constants of http://www.diva-portal.org/smash/get/diva2:1020192/FULLTEXT02.pdf drone
m = 1.07         #kg
Ixx = 0.0093   #kg-m^2
Iyy = 0.0092   #kg-m^2
# Izz = 0.9*(Ixx + Iyy) #kg-m^2 (Assume nearly flat object, z=0)
Izz = .0151
dx = 0.214     #m
# dy = 0.0825     #m
dy = dx
g = 9.81  #m/s/s
DTR = 1/57.3; RTD = 57.3
thrustCoef = 1.5108 * 10**-5 #kg*m

powerEst = []

bsRateLimiter = 0

u1 = []
u2 = []
u3 = []
u4 = []


def plotStuff(times, xdot_b, ydot_b, zdot_b, p, q, r, phi, theta, psi, x, y, z, u1, u2, u3, u4):
    df = pd.DataFrame(list(zip(times, xdot_b, ydot_b, zdot_b, p, q, r, phi, theta, psi, x, y, z)),
                      columns=['t', 'xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z'])

    plt.figure()
    plt.plot(times, u1, '.', label='u1')
    plt.plot(times, u2, '.', label='u2')
    plt.plot(times, u3, '.', label='u3')
    plt.plot(times, u4, '.', label='u4')
    plt.legend()
    plt.title('Body Frame of Reference')

    plt.figure()
    plt.plot(times, df['p'], '.', label='P')
    plt.plot(times, df['q'], '.', label='Q')
    plt.plot(times, df['r'], '.', label='R')
    plt.legend()
    plt.title('Body Frame of Reference')

    plt.figure()
    plt.plot(times, df['phi'], '.', label='phi')
    plt.plot(times, df['theta'], '.', label='theta')
    plt.plot(times, df['psi'], '.', label='psi')
    plt.legend()
    plt.title('External rotation')

    plt.figure()
    plt.plot(times, df['x'], '.', label='x')
    plt.title('X')

    plt.figure()
    plt.plot(times, df['y'], '.', label='y')
    plt.title('Y')

    plt.figure()
    plt.plot(times, df['z'], '.', label='z')
    plt.title('Z')

    plt.show()

    return df

class droneSim():
    def __init__(self, altControl, rollControl, pitchControl, yawControl, xControl, yControl, slowYawControl):

        self.xdot_b = []#latitudinal velocity body frame
        self.ydot_b = []#latitudinal velocity body frame
        self.zdot_b = []#latitudinal velocity body frame
        self.p = []#rotational velocity body frame
        self.q = []#rotational velocity body frame
        self.r = []#rotational velocity body frame
        self.phi = []#euler rotation global frame
        self.theta = []#euler rotation global frame
        self.psi = []#euler rotation global frame
        self.x = []#global x position
        self.y = []#global y position
        self.z = []#global z position

        self.altController = altControl
        self.rollController = rollControl
        self.pitchController = pitchControl
        self.yawController = yawControl

        self.xController = xControl
        self.yController = yControl
        self.slowYawController = slowYawControl

        self.rollSetPrev = 0
        self.pitchSetPrev = 0

        self.temp1 = []
        self.temp2 = []


    def stateMatrixInit(self):
        x = np.zeros(12)
        x[2] = -.049
        x[11] = .049
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

        modifyDef = 1000000   #initial Val = 10000000
        lesDef = .013385701848569465 * modifyDef

        for i,n in enumerate(u):
            try:
                w_o[i] = modifyDef *(-2/(1+np.e**((n/10)-5)) + 2) - lesDef #rough log equation mapping control signal (voltage) to rps
            except FloatingPointError as e:
                w_o[i] = modifyDef
            thrustForce[i] = thrustCoef * w_o[i]

        F1 = thrustForce[0] + thrustForce[2] + thrustForce[3]/2
        F2 = thrustForce[0] - thrustForce[1] - thrustForce[3]/2
        F3 = thrustForce[0] - thrustForce[2] + thrustForce[3]/2
        F4 = thrustForce[0] + thrustForce[1] - thrustForce[3]/2

        powerEst.append(thrustForce[0])

        return F1, F2, F3, F4

    def stateTransition(self, x, u):
        xdot = np.zeros(12)

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
        N = .01 * (F1 - F2 + F3 - F4) #.01 = drag coef?  random scaling for yaw

        # Pre-calculate trig values
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # Calculate the derivative of the state matrix using EOM
        try:
            xdot[0] = -g * sthe + r * vb - q * wb  # = udot
            xdot[1] = g * sphi * cthe - r * ub + p * wb  # = vdot
            xdot[2] = 1 / m * (-Fz) + g * cphi * cthe + q * ub - p * vb  # = wdot
            xdot[3] = 1 / Ixx * (L + (Iyy - Izz) * q * r)  # = pdot
            xdot[4] = 1 / Iyy * (M + (Izz - Ixx) * p * r)  # = qdot
            xdot[5] = 1 / Izz * (N + (Ixx - Iyy) * p * q)  # = rdot
            xdot[6] = p + (q * sphi + r * cphi) * sthe / cthe  # = phidot
            xdot[7] = q * cphi - r * sphi  # = thetadot
            xdot[8] = (q * sphi + r * cphi) / cthe  # = psidot
            xdot[9] = cthe * cpsi * xdot[0] + (-cthe * spsi + sphi * sthe * cpsi) * xdot[1] + \
                      (sphi * spsi + cphi * sthe * cpsi) * xdot[2]  # = xEdot
            xdot[10] = cthe * spsi * xdot[0] + (cphi * cpsi + sphi * sthe * spsi) * xdot[1] + \
                       (-sphi * cpsi + cphi * sthe * spsi) * xdot[2]  # = yEdot
            xdot[11] = -1 * (-sthe * xdot[0] + sphi * cthe * xdot[1] + cphi * cthe * xdot[2])  # = zEdot
        except Exception as e:
            print('tt')

        return xdot

    def numericalIntegration(self, x, errs, dt):
        # for now accept whatever we get from the derivative, maybe in future use Runge
        u = np.zeros(4)
        u[0] = self.altController.updateControl(errs[0])
        u[1] = self.rollController.updateControl(errs[1])
        u[2] = self.pitchController.updateControl(errs[2])
        u[3] = self.yawController.updateControl(errs[3])

        x_next = x + self.stateTransition(x, u) * dt

        return x_next, u

    def calculateError(self,x, setpoints):
        #store current errors?
        #setpoints = [alt, roll, pitch, yaw]
        altError= setpoints[0] - x[11]
        rollError= setpoints[1] - x[3]
        pitchError= setpoints[2] - x[4]
        yawError= setpoints[3] - x[5]

        return altError, rollError, pitchError, yawError

    def globalNeededThrust(self,x, u_x, u_y):
        #from https://liu.diva-portal.org/smash/get/diva2:1129641/FULLTEXT01.pdf, page 48

        phi = x[6]
        theta = x[7]
        psi = x[8]

        # Pre-calculate trig values
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # u_x = ((spsi*sphi) + (cpsi*cphi*sthe)) / (cthe*cpsi)
        # u_y = (-(cpsi*sphi) + (cpsi*cphi*sthe)) / (cthe*cpsi)

        maxAngleAllowed = 0.2745329 #10 degrees, actual max angle is ~36.86
        Kmax = np.sin(maxAngleAllowed)**2 / (1 - np.sin(maxAngleAllowed)**2)

        k = min(Kmax, u_x ** 2 + u_y ** 2)

        if u_x == 0 and u_y == 0:
            phiRef = 0
        else:
            phiRef = np.arcsin(np.sqrt(k/((1+k)*(u_x**2 + u_y**2)))*(u_x*spsi - u_y*cpsi))

        s = np.sign(u_y * spsi + u_x*cpsi)
        thetaRef = s*np.arccos(1/(np.cos(phiRef)*np.sqrt(1+k)))
        # thetaRef = 0

        return phiRef, thetaRef

    def controlInputs(self, x, t):
        # Inputs: Current state x[k], time t
        # Returns: Control inputs u[k]????

        globalXref = 5
        globalYref = 5



        if bsRateLimiter % 10 == 0:

            # currYaw = x[8]
            xError = x[9] - globalXref
            yError = x[10] - globalYref
            # xError = globalXref - x[9]
            # yError = globalYref - x[10]

            desiredXvel = self.xController.updateControl(xError)
            desiredYvel = self.yController.updateControl(yError)

            rollSet, pitchSet = self.globalNeededThrust(x, desiredXvel, desiredYvel)

            self.rollSetPrev = rollSet
            self.pitchSetPrev = pitchSet

        else:
            rollSet = self.rollSetPrev
            pitchSet = self.pitchSetPrev


        if t < 5:
            altSetpoint = 10
        elif 5<=t<10:
            altSetpoint = 2
        else:
            altSetpoint = 12

        rollSetpoint = rollSet
        pitchSetpoint = pitchSet
        yawSetpoint = 0

        self.temp1.append(rollSetpoint)
        self.temp2.append(pitchSetpoint)

        return altSetpoint, rollSetpoint, pitchSetpoint, yawSetpoint

    def randomWind(self, x):
        if np.random.uniform(0,100,1) > 99:
            dirxy = np.random.uniform(-np.pi, np.pi)
            dirz = np.random.uniform(-np.pi, np.pi)
            mag = (np.pi/5)*np.random.random(1)

            zmod = mag*np.cos(dirz)
            xmod = mag * np.sin(dirxy) * np.sin(dirz)
            ymod = mag * np.cos(dirxy) * np.sin(dirz)

            x[11] = x[11] + zmod
            x[3] = x[3] + xmod
            x[4] = x[4] + ymod

            return x
        else:
            return x


    def memory(self,x):
        self.xdot_b.append(x[0])
        self.ydot_b.append(x[1])
        self.zdot_b.append(x[2])
        self.p.append(x[3])
        self.q.append(x[4])
        self.r.append(x[5])
        self.phi.append(x[6])
        self.theta.append(x[7])
        self.psi.append(x[8])
        self.x.append(x[9])
        self.y.append(x[10])
        self.z.append(x[11])


if __name__ == "__main__":
    # altControl = PIDcontrol.PIDControl('Alt', Kp = 1, Ki = 0, Kd = 0, open = False)
    altControl = PIDcontrol.PIDControl('Alt', Kp = 15, Ki = .3, Kd = 20, open = False)
    rollControl = PIDcontrol.PIDControl('Roll', Kp = 2, Ki = .006, Kd = 20, open = False)
    pitchControl = PIDcontrol.PIDControl('Pitch', Kp = 4, Ki = .002, Kd = 9, open = False)
    yawControl = PIDcontrol.PIDControl('Yaw', Kp = 200, Ki = .02, Kd = 5, open = False)

    xControl = PIDcontrol.PIDControl('X', Kp = .015, Ki = .000001, Kd = 5, open = False)
    yControl = PIDcontrol.PIDControl('Y', Kp = .015, Ki = .000001, Kd = 5, open = False)
    slowYawControl = PIDcontrol.PIDControl('slowYaw', Kp = 2, Ki = .01, Kd = .02, open = True)

    ds = droneSim(altControl, rollControl, pitchControl, yawControl, xControl, yControl, slowYawControl)
    t = 0
    dt = .01
    times = []
    plt.ion()



    x = ds.stateMatrixInit()
    # x[3] = -.2
    # x[4] = .2
    # x[5] = .2

    while t < 20:
        times.append(t)
        t += dt
        # x = ds.randomWind(x)

        errs = ds.calculateError(x, ds.controlInputs(x,t))

        x_next, currU = ds.numericalIntegration(x, errs, dt)

        u1.append(currU[0])
        u2.append(currU[1])
        u3.append(currU[2])
        u4.append(currU[3])

        # Check for ground collision
        if x[11] < .05:
            x_next[2] = np.min([x[2],x_next[2],0])
            x_next[11] = np.max([x[11],x_next[11],0])

            x_next[3] = 0
            x_next[4] = 0
            x_next[5] = 0

        ds.memory(x_next)
        x = x_next


        # if t > 5:
        #     print('tt')

    df = plotStuff(times, ds.xdot_b, ds.ydot_b, ds.zdot_b, ds.p, ds.q, ds.r, ds.phi, ds.theta,ds.psi, ds.x, ds.y, ds.z, u1, u2, u3, u4)



    with open(r'C:\Users\Stephen\PycharmProjects\QuadcopterSim\visualizer\test.js', 'w') as outfile:
        outfile.truncate(0)
        outfile.write("var sim_data = [ \n")
        json.dump([i*2 for i in times], outfile, indent=4)
        outfile.write(",\n")
        json.dump([], outfile, indent=4)
        outfile.write(",\n")
        parsed = json.loads(
            df[['xdot_b', 'ydot_b', 'zdot_b', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']].T.to_json(
                orient='values'))
        json.dump(parsed, outfile, indent=4)
        outfile.write("]")

    print('tt')