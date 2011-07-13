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

import numpy as np
import scipy as sp
import sympy as sympy
from eng.material import elastic
from eng.composite import layup
from eng.composite import clt
from eng.composite import clpt
import copy
import sys

class Cylinder(clt.Laminate):
    #
    # Applied loading
    #
    
    axial_load = 0
    torsion = 0
    pressure_external = 0
    pressure_internal = 0
    delta_t = 0
    
    omega = 0
    
    radius = 1
    
    
    #
    # coordinates
    #

    x = sympy.Symbol('x')
    r = sympy.Symbol('r')

    def __init__(self):
        return
    
    def solve(self, inter_stresses=None):
        n = self.ply_count
        if inter_stresses is None:
            inter_stresses = np.zeros(n-1)
        
        # create containers for unknowns
        fmt_str = '%sn%0.0f'
        c_str = ['B','C','K1','K2']
        c_num = len(c_str)
        c_lambda = lambda i,j:sympy.Symbol(fmt_str % (c_str[j],i))
        c_mat = sympy.Matrix(n, c_num, c_lambda)
        self.unknowns = c_mat
        
        # container for stresses/strains/disp
        stress = sympy.Matrix(6, n, lambda i,j:0)
        strain = sympy.Matrix(6, n, lambda i,j:0)
        mech_strain = sympy.Matrix(6, n, lambda i,j:0)
        disp = sympy.Matrix(3, n, lambda i,j:0)
        
        
        R = self.radius
        x = self.x
        r = self.r
        
        P = 0
        T = 0
        #import pdb
        #pdb.set_trace()
        # get disp/strain/stress in each layer
        for i in xrange(self.ply_count):
            
            # gather constants to use
            Bn = c_mat[i,0]
            Cn = c_mat[i,1]
            K1n = c_mat[i,2]
            K2n = c_mat[i,3]
            
            # get ply geometry
            ri = self.convert_index_to_position(i) + R
            ro = self.convert_index_to_position(i+1) + R
            
            # get ply properties
            ply = self[i]
            a1 = ply.mat.coe['thermal',0]
            a2 = ply.mat.coe['thermal',1]
            a3 = ply.mat.coe['thermal',2]
            theta = ply.theta
            m = np.cos(theta)
            n = np.sin(theta)
            ax = m**2 * a1 + n**2 * a2
            at = n**2 * a1 + m**2 * a2
            ar = a3
            axt = 2 * n * m * a1 - 2 * n * m * a2
            
            cr = self[i].mat.cr
            cb = cr.rotate(theta,(0,0,1))
            #print cb
            
            # define capital sigma
            sig = ((cb[0,2] - cb[0,1]) * ax +
                   (cb[1,2] - cb[1,1]) * at +
                   (cb[2,2] - cb[2,1]) * ar +
                   (cb[5,2] - cb[5,1]) * axt) * self.delta_t
            
            n = np.sqrt(cb[1,1]/cb[2,2])
            
            # define displacements
            u = Bn * x
            v = Cn * x * r
            
            w = K1n * r**n + K2n * r**(-n)
            
            if (cb[2,2] == cb[1,1]):
                A1 = ((cb[0,1] - cb[0,2])*Bn + sig)/(2*cb[1,1])
                A2 = ((cb[1,5] - 2*cb[2,5])*Cn)/(4*cb[2,2]-cb[1,1])
                #A2 = ((cb[1,5] - cb[2,5])*Cn)/(4*cb[2,2]-cb[1,1])
                A3 = (-ply.mat.rho * self.omega**2)/(9*cb[2,2]-cb[1,1])
                w = w + (A1*r*sympy.log(r) + A2*r**2 + A3*r**3)
            elif (4*cb[2,2] == cb[1,1]):
                raise Exception('Not implemented case 2')
            elif (9*cb[2,2] == cb[1,1]):
                raise Exception('Not implemented case 3')
            else:
                A1 = ((cb[0,1] - cb[0,2])*Bn + sig)/(cb[2,2]-cb[1,1])
                A2 = ((cb[1,5] - 2*cb[2,5])*Cn)/(4*cb[2,2]-cb[1,1])
                #A2 = ((cb[1,5] - cb[2,5])*Cn)/(4*cb[2,2]-cb[1,1])
                A3 = (-ply.mat.rho * self.omega**2)/(9*cb[2,2]-cb[1,1])
                #import pdb; pdb.set_trace()
                w = w + (A1*r + A2*r**2 + A3*r**3)
            
            # define strains (these are engineering strains!)
            strain[:,i] = sympy.Matrix([sympy.simplify(sympy.diff(u,x)),
                                        sympy.simplify(w / r),
                                        sympy.simplify(sympy.diff(w,r)),
                                        sympy.simplify(1*(r*sympy.diff(v,r)-v)/r),
                                        sympy.simplify(1*sympy.diff(u,r)),
                                        sympy.simplify(1*sympy.diff(v,x))])
            
            alpha = sympy.Matrix([ax,at,ar,0,0,axt])
            
            mech_strain[:,i] = strain[:,i] - alpha*self.delta_t
            
            # define stress
            cbm = sympy.Matrix(cb.shape[0], cb.shape[1], lambda i,j: cb[i,j])
            stress[:,i] = cbm * mech_strain[:,i]
            
            # add to applied load/torque
            #import pdb; pdb.set_trace()
            # something in sympy doesn't like finding the integral of
            # Log[r] using the letter "r" specifically...works for "z"??
            from sympy.abc import z
            P_int = sympy.expand(stress[0,i]) * r
            P += 2*np.pi*sympy.integrate(P_int.expand().subs(r,z),(z,ri,ro)).subs(z,r)
            T_int = sympy.expand(stress[5,i]) * r**2
            T += 2*np.pi*sympy.integrate(T_int.expand().subs(r,z), (z,ri,ro)).subs(z,r)
            
            disp[:,i] = sympy.Matrix([u,v,w])
            # end of ply calculations

        #
        # Collect equations for BCs and continuity conditions
        #

        eqns = []
        n = self.ply_count
        
        # end tractions
        eqns.append(P - self.axial_load)
        eqns.append(T - self.torsion)

        # inner/outer tractions (shears already assumed zero)
        ri = R + self.convert_index_to_position(0)
        ro = R + self.convert_index_to_position(n)
        eqns.append(stress[2,0].subs(r, ri) - (-self.pressure_internal))
        eqns.append(stress[2,n-1].subs(r, ro) - (-self.pressure_external))
        
        #import pdb; pdb.set_trace()
        # continuity at layer interfaces
        for i in xrange(n-1):
            rint = R + self.convert_index_to_position(i+1)
            # u
            ui = disp[0,i]
            uo = disp[0,i+1]
            eqns.append((uo - ui).subs(r,rint).evalf())
            #v
            vi = disp[1,i]
            vo = disp[1,i+1]
            eqns.append((vo - vi).subs(r,rint).evalf())
            #w
            wi = disp[2,i]
            wo = disp[2,i+1]
            eqns.append((wo - wi).subs(r,rint).evalf())
            # sigma
            si = stress[2,i]
            so = stress[2,i+1]
            eqns.append((so - si + inter_stresses[i]).subs(r,rint).evalf())
            
        # create matrix from coeffs
        c_list = list(c_mat.reshape(c_num*n,1))
        c_zero = {}.fromkeys(c_list,0)
        unknown_count = c_num*n
        eqns_mat = sympy.Matrix(unknown_count, unknown_count, lambda i,j:0)
        rhs_vec = sympy.Matrix(unknown_count, 1, lambda i,j:0)
        for i in xrange(unknown_count):
            for j in xrange(self.ply_count):
                for k in xrange(c_num):
                    coef = eqns[i].evalf().coeff(c_mat[j,k])
                    unknown_id = c_num*j + k
                    if coef:
                        eqns_mat[i,unknown_id] = coef
                rhs_vec[i] = -eqns[i].subs(c_zero).evalf()
        eqns_mat = eqns_mat.subs(x,1)
        eqns_mat_np = np.zeros(eqns_mat.shape)
        rhs_vec_np = np.zeros(rhs_vec.shape)
        for i in xrange(eqns_mat.shape[0]):
            rhs_vec_np[i] = rhs_vec[i]
            for j in xrange(eqns_mat.shape[1]):
                eqns_mat_np[i,j] = eqns_mat[i,j]

        #
        # Solve for constants
        #

        # gather constants in flat list
        c_sol = {}
        
        sol = np.dot(np.linalg.inv(eqns_mat_np), rhs_vec_np)
        
        for i in xrange(len(c_list)):
            c_sol[c_list[i]] = sol[i,0]
        
        self.unknowns_solved = self.unknowns.subs(c_sol)
        self.unknowns_dict = c_sol
        
        #import pdb; pdb.set_trace()
        def create_piecewise(mat):
            rows = mat.shape[0]
            plys = mat.shape[1]
            pw = sympy.Matrix(rows, 1, lambda i,j:0)
            for i in xrange(rows):
                for j in xrange(plys):
                    ri = R + self.convert_index_to_position(j)
                    ro = R + self.convert_index_to_position(j+1)
                    pw[i] = sympy.Piecewise((pw[i],r<ri),(mat[i,j],r >= ri))
                pw[i] = sympy.Piecewise((pw[i],r <= ro),(0,True))
            return pw
        self.disp = create_piecewise(disp.subs(c_sol).evalf())
        self.strain = create_piecewise(strain.subs(c_sol).evalf())
        self.stress = create_piecewise(stress.subs(c_sol).evalf())
        return
    def plot_list(self, var, n=10):
        # the goal of this is to return a list of (w[r],r), given some
        # variable w[r] such as disp, strain, stress, etc.
        # n is the approximate points per ply
        rs = self.radius + self.linspace(n * self.ply_count)
        w_of_rs = self.linspace(n * self.ply_count) * 0
        for i in xrange(len(rs)):
            w_of_rs[i] = var.subs(self.r, rs[i])
        return (w_of_rs,rs)
        
