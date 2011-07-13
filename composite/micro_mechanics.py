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

from eng.material import elastic
from eng.composite import layup
from eng.composite import predefined
from eng.math import space
from eng.composite import kirchhoff
import numpy
import scipy

from abc import ABCMeta

class LaminateModel(object):
    '''Abstract Laminate Model
    I'd like this to be a true abstract class, but I'm not sure how to do
    that, so for now it just raises exceptions when needed.'''
    ## __metaclass__ = ABCMeta
    
    fiber = None
    matrix = None
    
    rf = 0
    rm = 1
    
    
    def __get_vf(self):
        vf = (self.rf**2)/(self.rm**2)
        return vf
    def __set_vf(self, value):
        rf = numpy.sqrt(value * self.rm**2)
        self.rf = rf
    vf = property(__get_vf, __set_vf)
    
    def __init__(self, fiber=None, matrix=None, vf=0):
        self.fiber = fiber
        self.matrix = matrix
        self.vf = vf
    
    def __get_e11(self):
        return self._get_e11()
    e11 = property(__get_e11)
    
    def __get_v12(self):
        return self._get_v12()
    v12 = property(__get_v12)
    
    def __get_coe(self):
        coe = elastic.CoefficientOfExpansion()
        for i in self.fiber.coe:
            coe[i,:] = self._get_coe(i)
        return coe
    coe = property(__get_coe)
    
    ## @abstractmethod
    def _get_e11(self):
        raise Exception('Abstract function not implemented')
    ## @abstractmethod
    def _get_v12(self):
        raise Exception('Abstract function not implemented')
    ## @abstractmethod
    def _get_coe(self, type):
        raise Exception('Abstract function not implemented')

class ConcentricCylindersModel(LaminateModel):
    
    def _get_e11(self):
        c11f = self.fiber.cr[0,0]
        c22f = self.fiber.cr[1,1]
        c12f = self.fiber.cr[0,1]
        c23f = self.fiber.cr[1,2]
        c11m = self.matrix.cr[0,0]
        c22m = self.matrix.cr[1,1]
        c12m = self.matrix.cr[0,1]
        c23m = self.matrix.cr[1,2]
        vf = self.vf
        e11 = (((-(c22f+c22m+c23f-c23m))*(-2*c12m**2+c11m*(c22m+c23m))-
                (-2*c12f**2*c22m+2*c12m**2*(c22f+2*c22m+c23f-2*c23m)-
                 2*c12f**2*c23m+4*c12f*c12m*(-c22m+c23m)+
                 c11f*(c22f+c22m+c23f-c23m)*(c22m+c23m)-
                 2*c11m*(c22m**2+(c22f+c23f-c23m)*c23m))*vf+
                (2*(c12f-c12m)**2-(c11f-c11m)*(c22f-c22m+c23f-c23m))*(c22m-c23m)*vf**2)/
               ((-(c22f+c22m+c23f-c23m))*(c22m+c23m)+
                (c22m-c23m)*(-c22f+c22m-c23f+c23m)*vf))
        return e11
    
    def _get_v12(self):
        ## c11f = self.fiber.cr[0,0]
        c22f = self.fiber.cr[1,1]
        c12f = self.fiber.cr[0,1]
        c23f = self.fiber.cr[1,2]
        ## c11m = self.matrix.cr[0,0]
        c22m = self.matrix.cr[1,1]
        c12m = self.matrix.cr[0,1]
        c23m = self.matrix.cr[1,2]
        vf = self.vf
        v12 = (((-c12m)*(c22f+c22m+c23f-c23m)*(-1+vf)+2*c12f*c22m*vf)/
               ((-c22m**2)*(-1+vf)-(c23f-c23m)*c23m*(-1+vf)+c22m*c23f*(1+vf)+
                c22f*(c22m+c23m+c22m*vf-c23m*vf)))
        
        return v12
    
    def _get_coe(self, type):
        c11f = self.fiber.cr[0,0]
        c22f = self.fiber.cr[1,1]
        c12f = self.fiber.cr[0,1]
        c23f = self.fiber.cr[1,2]
        c11m = self.matrix.cr[0,0]
        c22m = self.matrix.cr[1,1]
        c12m = self.matrix.cr[0,1]
        c23m = self.matrix.cr[1,2]
        vf = self.vf
        
        a1f = self.fiber.coe[type,0]
        a1m = self.matrix.coe[type,0]
        a2f = self.fiber.coe[type,1]
        a2m = self.matrix.coe[type,1]
        
        a1 = (((-a1m)*(-1+vf)*((c22f+c22m+c23f-c23m)*(-2*c12m**2+
                                                      c11m*(c22m+c23m))-
                               (c22m-c23m)*(2*(c12f-c12m)*c12m+
                                            c11m*
                                            (-c22f+c22m-c23f+c23m))*vf)+
               vf*(-2*(a2f-a2m)*(c22m-c23m)*
                   ((-c12m)*(c22f+c23f)+c12f*(c22m+c23m))*(-1+vf)+
                   a1f*(c11f*(c22f+c22m+c23f-c23m)*(c22m+c23m)+
                        2*c12f*c12m*(c22m-c23m)*(-1+vf)+
                        c11f*(c22m-c23m)*(c22f-c22m+c23f-c23m)*vf-
                        2*c12f**2*(c22m+c23m+c22m*vf-c23m*vf))))/
              (4*c12f*c12m*(c22m-c23m)*(-1+vf)*vf+
               2*c12m**2*(-1+vf)*(c22f+c22m+c23f+c23m*(-1+vf)-c22m*vf)-
               c11m*(-1+vf)*((c22f+c22m+c23f-c23m)*(c22m+c23m)+
                             (c22m-c23m)*(c22f-c22m+c23f-c23m)*vf)+
               vf*(c11f*(c22f+c22m+c23f-c23m)*(c22m+c23m)+
                   c11f*(c22m-c23m)*(c22f-c22m+c23f-c23m)*vf-
                   2*c12f**2*(c22m+c23m+c22m*vf-c23m*vf))))
        
        a2 = ((vf*((-a1f)*(2*c12f**2*c12m-2*c12f*(c12m**2-c11m*c22m)-
                           c11f*c12m*(c22f+c22m+c23f-c23m))*(-1+vf)+
                   a1m*(2*c12f**2*c12m-2*c12f*(c12m**2-c11m*c22m)-
                        c11f*c12m*(c22f+c22m+c23f-c23m))*(-1+vf)+
                   2*a2f*(c11m*c22m*(c22f+c23f)+
                          c12m**2*(c22f+c23f)*(-1+vf)+
                          c12f*c12m*(c22m-c23m)*(-1+vf)+
                          c22m*(-2*c12f**2+(c11f-c11m)*(c22f+c23f))*vf))-
               a2m*(-1+vf)*(2*c12m**2*(c22f+c22m+c23f-c23m)*(-1+vf)+
                            2*c12f*c12m*(-c22m+c23m)*vf-
                            (c22m+c23m)*(2*c12f**2*vf+
                                         (c22f+c22m+c23f-c23m)*
                                         (c11m*(-1+vf)-c11f*vf))))/
              (4*c12f*c12m*(c22m-c23m)*(-1+vf)*vf+
               2*c12m**2*(-1+vf)*(c22f+c22m+c23f+c23m*(-1+vf)-c22m*vf)-
               c11m*(-1+vf)*((c22f+c22m+c23f-c23m)*(c22m+c23m)+
                             (c22m-c23m)*(c22f-c22m+c23f-c23m)*vf)+
               vf*(c11f*(c22f+c22m+c23f-c23m)*(c22m+c23m)+
                   c11f*(c22m-c23m)*(c22f-c22m+c23f-c23m)*vf-
                   2*c12f**2*(c22m+c23m+c22m*vf-c23m*vf))))
        
        r = numpy.ndarray([1,6])
        r[0,0] = a1
        r[0,1] = a2
        
        return r
    
    

class RuleOfMixtures(LaminateModel):
    
    def _get_e11(self):
        e11f = self.fiber.e11
        e11m = self.matrix.e11
        vf = self.vf
        e11 = e11m*(1-vf)+e11f*vf
        return e11
    
    def _get_coe(self, type):
        e11f = self.fiber.e11
        e11m = self.matrix.e11
        vf = self.vf
        v12f = self.fiber.v12
        v12m = self.matrix.v12
        
        a1f = self.fiber.coe[type,0]
        a1m = self.matrix.coe[type,0]
        a2f = self.fiber.coe[type,1]
        a2m = self.matrix.coe[type,1]
        
        a1 = ((e11f*vf*a1f+e11m*(1-vf)*a1m)/(e11m*(1-vf)+e11f*vf))
        
        a2 = (vf*(-((e11m*v12f*(1-vf)*(-a1f+a1m))/
                    (e11m*(1-vf)+e11f*vf))+a2f)+
              (1-vf)*((e11f*v12m*vf*(-a1f+a1m))/
                      (e11m*(1-vf)+e11f*vf)+a2m))
        ret = numpy.ndarray([1,6])
        ret[0,0] = a1
        ret[0,1] = a2
        
        return ret
    

