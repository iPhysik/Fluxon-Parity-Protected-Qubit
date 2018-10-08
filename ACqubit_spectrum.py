# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 00:07:26 2018

@author: Wenyuan Zhang wzhang@physics.rutgers.edu

This program is to calculate energy spectrum for Aharonov-Casher Qubit. 
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import *
from scipy import optimize

from wavefunction import *
from wavefunction.wavefunction1d import *
from wavefunction.utils import *

def U_ho(x, args):
    """
    Harmonic oscillator potential
    """
    
    k = args['k']
    x0    = args['x0']
    
    u = 1/2 * k * ((x - x0) ** 2)

    return u

if __name__=='__main__':
    plt.close('all')
    h=1
    E_J =6.25
    E_C = 6.7
    E_L = 0.2
    E_CL = 5
    Delta_E_J =0.5
#    E_J=E_J-Delta_E_J
    
    phi_min,phi_max = [-15,15]
    N=20
    
    Q_dim=3
    Q_list = np.arange(Q_dim)
    ng_list = [1,1.5]
    NN=80
    NNN = NN * size(ng_list)
    spectrum01=np.zeros((NNN,2))
    spectrum02=np.zeros((NNN,2))
    
    Q = np.diag(Q_list).astype(np.complex)
                
    Q_ = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_[m,n]=mod_kron(m+1,n)+mod_kron(m-1,n)
                
    Q_p = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_p[m,n]=-1j*mod_kron(m+1,n)+1j*mod_kron(m-1,n)
    
        
    i=0

    for ng in ng_list:
        
        # charging energy of CPB
        H0 = 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(ones(N)) )
        
        
        for x0 in np.linspace(-2*np.pi,2*np.pi,NN):
            
            x = linspace(phi_min,phi_max,N)+x0
            args = {'k': E_L, 'x0': x0}
            U = U_ho(x, args);
            u = assemble_u_potential(N, U_ho, x, args)
            V = np.kron(Q,assemble_V(N,u))
            
            # kinetic energy of Superinductor
            K = np.kron (Q,assemble_K(N,-4*E_CL,phi_min+x0,phi_max+x0))
            H = K+V
            
            u = -E_J * np.cos(x/2)
            V = np.kron(Q_, assemble_V(N,u)) 
            H +=V
            
            # asymmetry in CPB
            if True:
                u = - 1./2 *Delta_E_J* np.sin(x/2)
                V = np.kron(Q_p,assemble_V(N,u)) 
                H += V 
                
#            H += 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(ones(N)) )
            H = H+H0

            evals, evecs = solve_eigenproblem(H)
            evals = evals.real
            
            spectrum01[i,:]=[x0,evals[1]-evals[0]]
            spectrum02[i,:]= [x0,evals[2]-evals[0]]
            i+=1
            #%%
    colors = plt.cm.jet(np.linspace(0.1,0.9,np.size(ng_list)))
    for i in np.arange(size(ng_list)):
        index = np.arange(i*NN,(i+1)*NN)
        plt.plot(spectrum01[index,0],spectrum01[index,1],'o-',c=colors[i],label='%.2f 01'%ng_list[i])
        plt.plot(spectrum02[index,0],spectrum02[index,1],'.-',c=colors[i],label='%.2f 02'%ng_list[i])
    plt.legend()
#        plt.figure()
#        plt.plot(energy0[:,0],energy0[:,1],'.-')
#        plt.plot(energy1[:,0],energy1[:,1],'.-')
#    plt.plot(spectrum02[:,0],spectrum02[:,1],'.-')

