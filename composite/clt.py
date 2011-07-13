""" Copyright (c) 2010, Tim Hartman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authored by: Tim Hartman"""

import numpy
import scipy
import sympy
from eng.material import elastic
from eng.composite import layup
from eng.math import space
import copy

class Ply(layup.Ply):
    """Ply of material.

    Longer class information for Ply.  Need to decide what goes here.

    Attributes:
        theta: material rotation (in radians).
    """
    
    def __get_q(self):
        return get_reduced_stiffness(self.mat.cr)
    q = property(__get_q)
    def __get_qbar(self):
        return get_reduced_stiffness_bar(self.q, self.theta)
    qbar = property(__get_qbar)
    
    

class Laminate(layup.Laminate):
    
    def __get_abd(self):
        abd = numpy.ndarray([6,6])
        abd.fill(0)
        for i in xrange(self.ply_count):
            abd += get_abd(self[i].qbar,
                           self.convert_index_to_position(i+0.5),
                           self[i].thickness)
        return abd
    abd = property(__get_abd)
    
    def get_piecewise_index(self, z = sympy.Symbol('z')):
        top = self.convert_index_to_position(0)
        pw = sympy.Piecewise((-1,z < top))
        # make a piecewise function for laminate qbars
        for i in xrange(len(self)):
            #messy, but works?
            bot = top
            top = self.convert_index_to_position(i+1)
            pw = sympy.Piecewise((pw,z < bot),(i, z <= top))
        pw = sympy.Piecewise((pw,z <= top),(-1, True))
        return pw
    

def get_reduced_stiffness(c):
    q11 = c[0,0] - (c[0,2]*c[0,2])/c[2,2]
    q12 = c[0,1] - (c[0,2]*c[1,2])/c[2,2]
    q22 = c[1,1] - (c[1,2]*c[1,2])/c[2,2]
    q66 = c[5,5]
    
    q = numpy.array([(q11, q12, 0), (q12, q22, 0), (0, 0, q66)])
    return q
    
def get_reduced_stiffness_bar(q, theta):
    m = numpy.cos(theta)
    n = numpy.sin(theta)
    
    m4 = m*m*m*m
    n4 = n*n*n*n
    m3 = m*m*m
    n3 = n*n*n
    m2 = m*m
    n2 = n*n
    
    qbar = numpy.ndarray([3,3])
    
    qbar[0,0] = q[0,0]*m4 + 2*(q[0,1]+2*q[2,2])*n2*m2 + q[1,1]*n4
    qbar[0,1] = (q[0,0]+q[1,1]-4*q[2,2])*m2*n2 + q[0,1]*(m4 + n4)
    qbar[1,0] = qbar[0,1]
    qbar[0,2] = (q[0,0]-q[0,1]-2*q[2,2])*m3*n + (q[0,1]-q[1,1]+2*q[2,2])*m*n3
    qbar[2,0] = qbar[0,2]
    qbar[1,1] = q[0,0]*n4 + 2*(q[0,1]+2*q[2,2])*n2*m2 + q[1,1]*m4
    qbar[1,2] = (q[0,0]-q[0,1]-2*q[2,2])*m*n3 + (q[0,1]-q[1,1]+2*q[2,2])*m3*n
    qbar[2,1] = qbar[1,2]
    qbar[2,2] = (q[0,0]+q[1,1]-2*q[0,1]-2*q[2,2])*m2*n2 + q[2,2]*(m4 + n4)
    
    return qbar

def get_abd(qbar, z0, thickness):
    z2 = z0 + thickness/2
    z1 = z0 - thickness/2
    
    t1 = (1/1.0)*(numpy.power(z2,1) - numpy.power(z1,1))
    t2 = (1/2.0)*(numpy.power(z2,2) - numpy.power(z1,2))
    t3 = (1/3.0)*(numpy.power(z2,3) - numpy.power(z1,3))
    
    t1_array = numpy.tile(t1, (3,3))
    t2_array = numpy.tile(t2, (3,3))
    t3_array = numpy.tile(t3, (3,3))
    
    ts = numpy.ndarray([6,6])
    ts[0:3,0:3] = t1
    ts[0:3,3:6] = t2
    ts[3:6,0:3] = t2
    ts[3:6,3:6] = t3
    
    qbars = numpy.tile(qbar,(2,2))
    
    abd = ts * qbars
    
    return abd





























