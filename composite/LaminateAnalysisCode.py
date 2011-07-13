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


print "Laminate Analysis Code"
print "(c) 2010, Tim Hartman\n"

from eng.material import elastic
from eng.composite import layup
from eng.composite import predefined
from eng.math import space
from eng.composite import kirchhoff
import scipy
import numpy

# define composite layup
lam = predefined.lac_test_02()

# define parameters
forces_applied = numpy.ndarray([6])
#forces_applied[:] = ( 150e3,
#                     -120e3,
#                      -68e3,
#                       12.5,
#                      -24.5,
#                        2.5)
forces_applied[:] = (1,0,0,0,0,0)
delta_t = 0
delta_m = 0
midsurface = numpy.ndarray([6])
midsurface[:] = (0,0,0,0,0,0)


def calc_from_strains():
    pass

def calc_strains_from_forces():
    forces_thermal = numpy.zeros(6)
    forces_moisture = numpy.zeros(6)
    
    for i in xrange(lam.ply_count):
        m = numpy.cos(lam[i].theta)
        n = numpy.sin(lam[i].theta)
        
        # assumes coe defined as (a1,a2) (transversely isotropic definition)
        for j in iter(lam[i].mat.coe):
            if j.name == 'thermal':
                alpha1 = j[0]
                alpha2 = j[1]
            elif j.name == 'moisture':
                beta1 = j[0]
                beta2 = j[1]
        
        a_x = alpha1*m*m + alpha2*n*n
        a_y = alpha1*n*n + alpha2*m*m
        a_xy = 2*(alpha1 - alpha2)*m*n
        
        b_x = beta1*m*m + beta2*n*n
        b_y = beta1*n*n + beta2*m*m
        b_xy = 2*(beta1 - beta2)*m*n
        
        qb11 = lam[i].qbar[0,0]
        qb12 = lam[i].qbar[0,1]
        qb16 = lam[i].qbar[0,2]
        qb22 = lam[i].qbar[1,1]
        qb26 = lam[i].qbar[1,2]
        qb66 = lam[i].qbar[2,2]
        
        z0 = lam.convert_index_to_position(i+0.5)
        z_bot = z0 - lam[i].thickness/2
        z_top = z0 + lam[i].thickness/2
            
        t1 = (1/1.0)*(numpy.power(z_top,1) - numpy.power(z_bot,1))
        t2 = (1/2.0)*(numpy.power(z_top,2) - numpy.power(z_bot,2))
        
        forces_thermal += ((qb11*a_x + qb12*a_y + qb16*a_xy)*delta_t*t1,
                           (qb12*a_x + qb22*a_y + qb26*a_xy)*delta_t*t1,
                           (qb16*a_x + qb26*a_y + qb66*a_xy)*delta_t*t1,
                           (qb11*a_x + qb12*a_y + qb16*a_xy)*delta_t*t2,
                           (qb12*a_x + qb22*a_y + qb26*a_xy)*delta_t*t2,
                           (qb16*a_x + qb26*a_y + qb66*a_xy)*delta_t*t2)
        
        forces_moisture += ((qb11*b_x + qb12*b_y + qb16*b_xy)*delta_m*t1,
                            (qb12*b_x + qb22*b_y + qb26*b_xy)*delta_m*t1,
                            (qb16*b_x + qb26*b_y + qb66*b_xy)*delta_m*t1,
                            (qb11*b_x + qb12*b_y + qb16*b_xy)*delta_m*t2,
                            (qb12*b_x + qb22*b_y + qb26*b_xy)*delta_m*t2,
                            (qb16*b_x + qb26*b_y + qb66*b_xy)*delta_m*t2)
    
    forces = forces_applied + forces_thermal + forces_moisture
    
    small_abd = numpy.linalg.inv(lam.abd)
    
    strains = numpy.dot(small_abd, forces)
    
    return strains

strains = calc_strains_from_forces()









