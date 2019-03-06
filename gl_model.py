import pygame
import cv2
import scipy.misc

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np


# the range of this set is 60-120 -180-180 -30-30
class gl_ob(object):

    def __init__(self, width, height, batch_size, mode = 'random'):
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.mode = mode
        self.if_start = True
        self.scale = 3.5

        self.colors = ((1, 0, 0),
                       (0, 1, 0),
                       (0, 0, 1),
                       (1, 1, 0),
                       (1, 0, 1),
                       (0, 1, 1),
                       (1, .5, 0),
                       (1, 0, .5),
                       (.5, 1, 0),
                       (0, 1, .5),
                       (.5, 0, 1),
                       (0, .5, 1),
                       (1, 1, 0.5),
                       (1, 0.5, 1),
                       (0.5, 1, 1),
                       )

        self.model = self.loadModel()

    def cube(self):

        glBegin(GL_TRIANGLES)
        tri_num = len(self.model)/3

        for i in range(tri_num):

            tri = self.model[i*3:(i+1)*3]

            if i >=0 and i<4 or i>=80 and i <82 or i >= 96 and i < 98:
                glColor3fv(self.colors[0])
            elif i>=4 and i<80:
                glColor3fv(self.colors[1])
            elif i >= 82 and i < 88 or i>=142 and i<160 or i >= 90 and i < 96 or i >= 124 and i < 142:
                glColor3fv(self.colors[2])
            elif i >= 88 and i < 90:
                glColor3fv(self.colors[3])
            elif i >= 98 and i < 100:
                glColor3fv(self.colors[3])
            elif i >= 100 and i < 102:
                glColor3fv(self.colors[0])
            elif i >= 102 and i < 108:
                glColor3fv(self.colors[4])
            elif i >= 108 and i < 112:
                glColor3fv(self.colors[5])
            elif i >= 112 and i < 116:
                glColor3fv(self.colors[6])
            elif i >= 116 and i < 124:
                glColor3fv(self.colors[10])
            elif i >= 160 and i < 162:
                glColor3fv(self.colors[11])
            elif i >= 162 and i < 164:
                glColor3fv(self.colors[12])
            elif i >= 164 and i < 166:
                glColor3fv(self.colors[13])
            elif i >= 166 and i < 168:
                glColor3fv(self.colors[14])
            elif i >= 168 and i < 170:
                glColor3fv(self.colors[10])
            elif i >= 170 and i < 172:
                glColor3fv(self.colors[11])
            else:
                glColor3fv((1,1,1))


            for p in tri:
                glVertex3fv(p)

        glEnd()

    def loadModel(self):

        scale = 30

        point = []
        with open('./binary.stl', 'r') as f:

            index = 0

            while True:
                p = f.readline().strip()
                if p == 'endsolid':
                    break
                word = p.split()
                if word[0] == 'vertex':
                    x = scale*float(word[1])
                    y = scale*float(word[2])
                    z = scale*float(word[3])
                    point.append(None)
                    point[index] = (x, y, z)
                    index = index + 1

        return point


    def initiate(self):

        pygame.init()
        self.display = (int(self.height*self.scale), int(self.height*self.scale))
        self.window = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)

        gluPerspective(45, (float(self.display[0]) / float(self.display[1])), 0.1, 50.0)

        glTranslatef(0.0, 0.0, -5)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)



    def draw_ob(self, if_end=False):

        if self.if_start == True:
            self.initiate()
            self.if_start = False

        batch_example = np.zeros((self.batch_size,self.height,self.width))

        batch_p = [None]
        start_r1 = 0
        start_r2 = 0
        start_r3 = 0
        start_x = 0
        start_y = 0
        start_z = 0

        index = 0
        x_aix = np.random.uniform(-1, 1)
        y_aix = np.random.uniform(-1, 1)
        z_aix = np.random.uniform(-1, 1)
        ranx =0.25
        rany =0.25
        ranz= 0.25

        seed = 2019

        while True:


            if self.mode == 'random':

                glRotatef(0.5, x_aix, y_aix, z_aix)
                # glTranslatef(0,0,0.01)

            elif self.mode == 'random_xyz':
                if index == 0:
                    start_x = np.random.uniform(-ranx, ranx)
                    glTranslatef(start_x, 0, 0)
                    start_y = np.random.uniform(-rany, rany)
                    glTranslatef(0, start_y, 0)
                    start_z = np.random.uniform(-ranz, ranz)
                    glTranslatef(0, 0, start_z)
                    start_r3 = np.random.uniform(60,120)
                    glRotatef(start_r3, 1, 0, 0)
                    start_r2 = np.random.uniform(-180,180)
                    glRotatef(start_r2, 0, 1, 0)
                    start_r1 = np.random.uniform(-30,30)
                    glRotatef(start_r1, 0, 0, 1)

                    batch_p[0] = [start_r1,start_r2,start_r3,start_x,start_y,start_z]

                else:

                    glRotatef(-start_r1, 0, 0, 1)
                    glRotatef(-start_r2, 0, 1, 0)
                    glRotatef(-start_r3, 1, 0, 0)
                    glTranslatef(-start_x, 0, 0)
                    glTranslatef(0, -start_y, 0)
                    glTranslatef(0, 0, -start_z)

                    start_x = np.random.uniform(-ranx, ranx)
                    glTranslatef(start_x, 0, 0)
                    start_y = np.random.uniform(-rany, rany)
                    glTranslatef(0, start_y, 0)
                    start_z = np.random.uniform(-ranz, ranz)
                    glTranslatef(0, 0, start_z)
                    start_r3 = np.random.uniform(60,120)
                    glRotatef(start_r3, 1, 0, 0)
                    start_r2 = np.random.uniform(-180,180)
                    glRotatef(start_r2, 0, 1, 0)
                    start_r1 = np.random.uniform(-30,30)
                    glRotatef(start_r1, 0, 0, 1)


                    batch_p.append(None)
                    batch_p[index] = [start_r1,start_r2,start_r3,start_x,start_y,start_z]

            elif self.mode == 'random_xyz_with_seed':

                if index == 0:
                    np.random.seed(seed)
                    start_x = np.random.uniform(-ranx, ranx)
                    glTranslatef(start_x, 0, 0)
                    seed = seed+1
                    np.random.seed(seed)
                    start_y = np.random.uniform(-rany, rany)
                    glTranslatef(0, start_y, 0)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_z = np.random.uniform(-ranz, ranz)
                    glTranslatef(0, 0, start_z)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_r3 = np.random.uniform(60,120)
                    glRotatef(start_r3, 1, 0, 0)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_r2 = np.random.uniform(-180,180)
                    glRotatef(start_r2, 0, 1, 0)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_r1 = np.random.uniform(-30,30)
                    glRotatef(start_r1, 0, 0, 1)

                    batch_p[0] = [start_r1,start_r2,start_r3,start_x,start_y,start_z]

                else:

                    glRotatef(-start_r1, 0, 0, 1)
                    glRotatef(-start_r2, 0, 1, 0)
                    glRotatef(-start_r3, 1, 0, 0)
                    glTranslatef(-start_x, 0, 0)
                    glTranslatef(0, -start_y, 0)
                    glTranslatef(0, 0, -start_z)

                    seed = seed + 1
                    np.random.seed(seed)
                    start_x = np.random.uniform(-ranx, ranx)
                    glTranslatef(start_x, 0, 0)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_y = np.random.uniform(-rany, rany)
                    glTranslatef(0, start_y, 0)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_z = np.random.uniform(-ranz, ranz)
                    glTranslatef(0, 0, start_z)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_r3 = np.random.uniform(60,120)
                    glRotatef(start_r3, 1, 0, 0)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_r2 = np.random.uniform(-180,180)
                    glRotatef(start_r2, 0, 1, 0)
                    seed = seed + 1
                    np.random.seed(seed)
                    start_r1 = np.random.uniform(-30,30)
                    glRotatef(start_r1, 0, 0, 1)


                    batch_p.append(None)
                    batch_p[index] = [start_r1,start_r2,start_r3,start_x,start_y,start_z]

            elif self.mode == 'regular_xyz':

                if index == 0:
                    start_x = -ranx + 2*ranx * index/float(self.batch_size)
                    glTranslatef(start_x, 0, 0)
                    start_y = -rany + 2*rany * index/float(self.batch_size)
                    glTranslatef(0, start_y, 0)
                    start_z = -ranz + 2*ranz * index/float(self.batch_size)
                    glTranslatef(0, 0, start_z)
                    start_r3 = 60+60*index/float(self.batch_size)
                    glRotatef(start_r3, 1, 0, 0)
                    start_r2 = -180+360 * index / float(self.batch_size)
                    glRotatef(start_r2, 0, 1, 0)
                    start_r1 = -30+60 * index / float(self.batch_size)
                    glRotatef(start_r1, 0, 0, 1)

                    batch_p[0] = [start_r1, start_r2, start_r3, start_x, start_y, start_z]

                else:

                    glRotatef(-start_r1, 0, 0, 1)
                    glRotatef(-start_r2, 0, 1, 0)
                    glRotatef(-start_r3, 1, 0, 0)
                    glTranslatef(-start_x, 0, 0)
                    glTranslatef(0, -start_y, 0)
                    glTranslatef(0, 0, -start_z)

                    start_x = -ranx + 2 * ranx * index / float(self.batch_size)
                    glTranslatef(start_x, 0, 0)
                    start_y = -rany + 2 * rany * index / float(self.batch_size)
                    glTranslatef(0, start_y, 0)
                    start_z = -ranz + 2 * ranz * index / float(self.batch_size)
                    glTranslatef(0, 0, start_z)
                    start_r3 = 60 + 60 * index / float(self.batch_size)
                    glRotatef(start_r3, 1, 0, 0)
                    start_r2 = -180+360 * index / float(self.batch_size)
                    glRotatef(start_r2, 0, 1, 0)
                    start_r1 = -30+60 * index / float(self.batch_size)
                    glRotatef(start_r1, 0, 0, 1)

                    batch_p.append(None)
                    batch_p[index] = [start_r1, start_r2, start_r3, start_x, start_y, start_z]

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.cube()

            string_image = pygame.image.tostring(self.window, 'RGB')
            temp_surf = pygame.image.fromstring(string_image, self.display, 'RGB')
            tmp_arr = pygame.surfarray.array3d(temp_surf)
            # tmp_arr_gray = cv2.cvtColor(tmp_arr, cv2.COLOR_RGB2GRAY).T
            canny_im = cv2.Canny(tmp_arr,100,200).T
            kernel = np.ones((5,5), np.uint8)
            dilation = cv2.dilate(canny_im,kernel,iterations = 1)

            small = cv2.resize(dilation, (0, 0), fx=1/float(self.scale), fy=1/float(self.scale))

            num=0

            for i in range(self.width):
                for j in range(self.height):
                    if small[i,j] >0:
                        small[i,j] = 255

            batch_example[index] = small

            pygame.display.flip()
            pygame.time.wait(5)
            index = index + 1
            if index >= self.batch_size:
                glRotatef(-start_r1, 0, 0, 1)
                glRotatef(-start_r2, 0, 1, 0)
                glRotatef(-start_r3, 1, 0, 0)
                glTranslatef(-start_x, 0, 0)
                glTranslatef(0, -start_y, 0)
                glTranslatef(0, 0, -start_z)
                return batch_example, np.array(batch_p)




    def show_example(self):

        example, example_y = self.draw_ob()

        for i in range(self.batch_size):
            im = example[i]
            scipy.misc.imsave('./sample/{}.jpg'.format(i), im)

    def shut_down(self):
        pygame.quit()