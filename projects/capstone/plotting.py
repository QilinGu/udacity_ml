#
# Reinforcement learning agent for the "driverless car" using Tensorflow
#

import tensorflow as tf
import numpy as np
from learning import *
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

class Plotter:

    # Shallow Q-Learning network, as opposed to "deep"
    
    def __init__(self, name='q_value', track=False):
        # Tf graph input
        self.ai = Learner(True)
        self.saver = tf.train.Saver()
        self.saver.restore(self.ai.s,"models/narrow-deep-pipe.ckpt")
        self.theta = 0
        self.im = None
        self.fig = None
        self.sensor = 0
        self.smoothing = True
        matplotlib.rcParams['xtick.direction'] = 'out'
        matplotlib.rcParams['ytick.direction'] = 'out'

    def moving_average_2d(self, data, window):
        """Moving average on two-dimensional data.
        """
        # Makes sure that the window function is normalized.
        window /= window.sum()
        # Makes sure data array is a numpy array or masked array.
        if type(data).__name__ not in ['ndarray', 'MaskedArray']:
            data = numpy.asarray(data)

        # The output array has the same dimensions as the input data 
        # (mode='same') and symmetrical boundary conditions are assumed
        # (boundary='symm').
        return convolve2d(data, window, mode='same', boundary='symm')

    def location_contours(self, sensors=[0.2,0.2,0.2,0.0]):
        x = np.arange(0,1,0.01)
        y = np.arange(0,1,0.01)
        qt = self.ai.q_train
        a,b = np.meshgrid(x,y)
        s1,s2,s3,theta = sensors
        X = np.concatenate([[[s1, s2, s3, a[:,i][j], b[:,i][j], theta] for i in range(len(a[0]))] for j in range(len(a))])
        feed = {qt.x: X, qt.q_max: qt.q_max_val}
        q = self.ai.s.run(tf.reduce_max(qt.q_value, reduction_indices=[1]), feed_dict=feed)
        #q = self.ai.s.run(tf.argmax(qt.q_value, dimension=1), feed_dict=feed)
        cols = len(a[0])
        rows = len(a)
        c = np.array([[q[i*cols+j] for j in range(cols)] for i in range(rows)])
        if self.smoothing:
            c = self.moving_average_2d(c, np.ones((6,40)))
        return a,b,c

    def angle_v_sensor_contours(self, x0=0.5, y0=0.5):
        # x axis varies theta from 0 to 2*pi
        # y axis varies sensors all from 0 to 1.0 in unison
        x = np.arange(0,1,0.01)
        y = np.arange(0,1,0.01)
        qt = self.ai.q_train
        a,b = np.meshgrid(x,y)
        X = np.concatenate([[[b[:,i][j], b[:,i][j], b[:,i][j], x0, y0, 2*np.pi*a[:,i][j]] for i in range(len(a[0]))] for j in range(len(a))])
        feed = {qt.x: X, qt.q_max: qt.q_max_val}
        q = self.ai.s.run(tf.reduce_max(qt.q_value, reduction_indices=[1]), feed_dict=feed)
        #q = self.ai.s.run(tf.argmax(qt.q_value, dimension=1), feed_dict=feed)
        cols = len(a[0])
        rows = len(a)
        c = np.array([[q[i*cols+j] for j in range(cols)] for i in range(rows)])
        if self.smoothing:
            c = self.moving_average_2d(c, np.ones((6,40)))
        return a,b,c

    def contour_plot(self, sensors=[0.2,0.2,0.2,0.0], title="Contour Plot of Q(s,a)"):
        x,y,z = self.location_contours(sensors)
        plt.figure(facecolor='white')
        plt.hot()
        im = plt.imshow(z, interpolation='bilinear', origin='lower', cmap=cm.inferno)
        CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
        plt.title(title+": theta="+str(int(sensors[3]*180.0/np.pi)))
        plt.xlabel('x%')
        plt.ylabel('y%')
        plt.show()

    def angle_v_sensor_plot(self, x0=0.5, y0=0.5, title="Contour Plot of Q(s,a)"):
        x,y,z = self.angle_v_sensor_contours(x0, y0)
        plt.figure(facecolor='white')
        plt.hot()
        plt.xlabel('Orientation')
        plt.ylabel('Signal strength')
        im = plt.imshow(z, interpolation='bilinear', origin='lower', cmap=cm.inferno)
        CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
        plt.title(title)
        plt.show()

    def update_theta(self, *args):
        self.theta += np.pi/20.0
        x,y,z = self.location_contours([0.2, 0.2, 0.2, self.theta])
        self.theta %= (np.pi*2.0)
        self.im.set_data(z)
        self.fig.suptitle("Countour Q plot - Heading "+str(int(self.theta*180.0/np.pi)))
        return self.im

    def theta_anim(self):
        self.theta = 0
        x,y,z = self.location_contours([0.2, 0.2, 0.2, self.theta])
        self.fig = plt.figure()
        self.im = plt.imshow(z, interpolation='bilinear', origin='lower', cmap=cm.inferno)
        CBI = plt.colorbar(self.im, orientation='horizontal', shrink=0.8)
        plt.title('Contour Plot - Q')
        ani = animation.FuncAnimation(self.fig, self.update_theta, interval=50, blit=False)
        plt.show()

    def theta_gif(self):
        self.theta = 0
        x,y,z = self.location_contours([0.2, 0.2, 0.2, self.theta])
        self.fig = plt.figure()
        self.im = plt.imshow(z, interpolation='bilinear', origin='lower', cmap=cm.inferno)
        CBI = plt.colorbar(self.im, orientation='horizontal', shrink=0.8)
        plt.xlabel('X %')
        plt.ylabel('Y %')
        ani = animation.FuncAnimation(self.fig, self.update_theta, frames=np.arange(0,20), interval=200, blit=False)
        ani.save('figures/theta.gif', dpi=80, writer='imagemagick')

    def update_sensor(self, *args):
        self.theta += np.pi/20.0
        self.sensor += 0.02
        if self.sensor > 1:
            self.sensor = 0.0
        s = self.sensor
        x,y,z = self.location_contours([s, s, s, self.theta])
        self.im.set_data(z)
        self.fig.suptitle("Countour Q plot - Sensor "+str(self.sensor))
        return self.im

    def sensor_anim(self, theta=0):
        self.theta = theta
        self.sensor = 0.0
        x,y,z = self.location_contours([0,0,0, self.theta])
        self.fig = plt.figure()
        self.im = plt.imshow(z, interpolation='bilinear', origin='lower', cmap=cm.inferno)
        CBI = plt.colorbar(self.im, orientation='horizontal', shrink=0.8)
        ani = animation.FuncAnimation(self.fig, self.update_sensor, interval=50, blit=False)
        plt.show()


p = Plotter()