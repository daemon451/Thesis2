import sys
sys.path.append('..')
import metrics_v2
import geoUtilities
from geoUtilities import geoMove_Radians

import numpy as np
import os
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


sensorData = {'sensors': []}

iterations = 3000
numTracks = 1
numSensorsAve = 5
minSpeed = 2
maxSpeed = 8

class Ship():
    def __init__(self, time, id):
        self.maxspeed = np.random.uniform(minSpeed,maxSpeed) #meters per second
        self.track = id
        self.lat = 0
        self.lon = 0
        self.tolBearing = np.random.uniform(-np.pi, np.pi)
        self.tolVelocity = np.random.uniform(-self.maxspeed, self.maxspeed)
        self.tolDuration = np.random.uniform(500, 4000)
        self.tolStartTime = time
        self.currTime = self.tolStartTime
        self.lastTime = self.tolStartTime
        self.newTimeOnLeg()
        self.brokenFlag = 0

        self.zigzagTime = self.tolStartTime

    def newTimeOnLeg(self):
        self.tolStartTime = self.lastTime
        self.tolBearing = np.random.uniform(-np.pi, np.pi)
        self.tolVelocity = np.random.uniform(-self.maxspeed, self.maxspeed)
        self.tolDuration = np.random.uniform(600, 1500)

        self.newSensors()

    def newSensors(self):
        sensor1 = Sensor()
        sensor2 = Sensor()
        sensor3 = Sensor()
        sensor4 = Sensor()
        sensor5 = Sensor()

        self.sensors = {'sensor1':sensor1, 'sensor2':sensor2,'sensor3':sensor3,'sensor4':sensor4,'sensor5':sensor5}
        self.sensorLookup = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']

    def updatePosition(self, lat, lon, bearing, delta):



        return lat, lon


    def update(self,newTime, instanceId):
        if newTime > self.tolStartTime + self.tolDuration:
            self.newTimeOnLeg()


        #MAIN MOVEMENT MODEL
        (lat, lon) = self.updatePosition(self.lat, self.lon, self.tolBearing, self.tolVelocity * (newTime-self.lastTime))
        self.lat = lat
        self.lon = lon
        self.lastTime = newTime

        #pick a random sensor for this track to generate a report
        sense = np.random.randint(0,5)
        self.sensor = self.sensors[self.sensorLookup[sense]]
        xmeas, ymeas, self.outside = self.sensor.generateReportEllipse()

        bearingShift = np.arctan(xmeas/ymeas)
        bearingShift = (bearingShift + 5*np.pi/2) % 2*np.pi
        distanceShift = np.sqrt(xmeas**2 + ymeas**2)
        (self.measLat, self.measLon) = geoMove_Radians(self.lat, self.lon, bearingShift, distanceShift)

        debug = {"id": self.track, "lat":lat,"lon": lon,"timestamp": newTime*1000}
        jsonObj = {"id":instanceId, 'Lat':self.measLat,"Lon":self.measLon, "smaj":self.sensor.smaj/1852, "smin":self.sensor.smin/1852,
                       "rotation": self.sensor.rotate, "timestamp":newTime*1000, "source_id":self.sensor.id,
                       "bearing":0, "range":0,"DEBUG":debug}


        x,y,z = geoUtilities.geoToCart(lat, lon)
        a = np.max([self.sensor.smaj, self.sensor.smin])
        b = np.min([self.sensor.smaj, self.sensor.smin])

        return jsonObj


    def addmeterstogeo(self, lat, long, latMeters, longMeters):
        cart = geoUtilities.geoToCart(lat, long)
        cart = (cart[0] + longMeters, cart[1] + latMeters, cart[2])
        geo = geoUtilities.cartToGeo(cart[0], cart[1], cart[2])
        return geo[0], geo[1]


class Sensor():
    class_counter = count(0)
    def __init__(self):
        self.id = next(Sensor.class_counter)
        self.smajMax = np.random.uniform(10, 1000)
        self.sminMax = np.random.uniform(10, 500)
        self.rotation = np.random.uniform(-np.pi, np.pi)
        self.smaj = self.smajMax
        self.smin = self.sminMax
        self.containment = []

        self.sensorType = 0

        self.scalaMimicEllipse()
        self.biasSensor()

    def scalaMimicEllipse(self):
        self.smajMax = np.abs(np.random.uniform(1000,25000))
        self.sminMax = np.abs(np.random.uniform(1000,17200))


        if self.sminMax>self.smajMax:
            tt = self.smajMax
            self.smajMax = self.sminMax
            self.sminMax = tt


        self.rotate = np.random.uniform(0,np.pi)

        rho = np.sqrt(np.random.uniform(0, 1))
        phi = np.random.uniform(0, 2 * np.pi)
        x_c = rho * np.cos(phi)
        y_c = rho * np.sin(phi)
        x_e = x_c * self.smaj
        y_e = y_c * self.smin
        self.x = x_e * np.cos(self.rotate) - y_e * np.sin(self.rotate)
        self.y = x_e * np.sin(self.rotate) + y_e * np.cos(self.rotate)

    def biasSensor(self):
        alpha = self.rotation - np.pi/2

        theta = np.random.uniform(-np.pi, np.pi)
        maxR = self.smaj*self.smin/(np.sqrt((self.smaj*np.cos(theta))**2 + (self.smin*np.sin(theta))**2))
        rRange = [maxR*.9, maxR* 1.1]
        if rRange[1] > self.smaj or rRange[1] >self.smin:
            rRange[1] = np.max([self.smaj, self.smin])
        r = np.random.uniform(0, maxR)
        x = r*np.cos(theta + alpha)
        y1 = r*np.sin(theta + alpha)

        self.x = x
        self.y = y1


    def ellipseEnclosure(self,x, y):
        #V2 - works?
        xc = 0 - x
        yc = 0 - y
        g_ell_width = self.smaj
        g_ell_height = self.smin
        xct = xc * np.cos(np.pi - self.rotate) - yc * np.sin(np.pi - self.rotate)
        yct = xc * np.sin(np.pi - self.rotate) + yc * np.cos(np.pi - self.rotate)
        rad_cc = (xct ** 2 / (g_ell_width) ** 2) + (yct ** 2 / (g_ell_height) ** 2)
        self.containment.append(rad_cc)

        # ax = confidence_ellipse(0, 0, x, y, self.smaj, self.smin, self.rotate, title=self.id)

        return rad_cc


    def generateReportEllipse(self):

        #generate unique posit ellipse for this report
        self.smin = np.random.normal(self.sminMax,self.sminMax/10, 1)[0]
        self.smaj = np.random.normal(self.smajMax, self.smajMax/10, 1)[0]

        self.rotate = np.random.normal(self.rotation, np.pi/120,1)[0]

        self.xmeas = np.random.normal(self.x, np.divide(np.abs(self.smaj),10), 1)[0]
        self.ymeas = np.random.normal(self.y, np.divide(np.abs(self.smaj),10), 1)[0]

        outside = 0
        if self.ellipseEnclosure(self.xmeas, self.ymeas)>1:
            outside = 1

        return self.xmeas, self.ymeas, outside

if __name__ == "__main__":
    time = 0
    instanceId = 0
    availableTracks = []
    ships = {}

    out = 0
    plt.ion()

    with open('sensor_pythonSimData.txt', 'w') as outputFile:
        for i in range(0, iterations):
            time = time + np.random.uniform(0, 60)  #data rate of simulator * number of tracks
            track = "track_" + str(np.random.randint(0,numTracks))
            if track not in availableTracks:
                availableTracks.append(track)
                ships[track] = Ship(time, track)
                report = ships[track].update(time + 1, instanceId)
            else:
                report = ships[track].update(time, instanceId)
                out += ships[track].outside

            instanceId += 1
            outputFile.write(str(report))
            outputFile.write("\n")
        outputFile.close()
        print(out/iterations)
        if os.path.exists('sensor_pythonSimData.csv'):
            os.remove('sensor_pythonSimData.csv')


    print('tt')