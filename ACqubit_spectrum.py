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
    h=1
    E_CL = 5
    # E_CL = 1/8./4/pi**2
    E_L = 0.4
    # E_L = 4*pi**2
    E_J =6.25
    E_C = 6.7
    
    phi_min,phi_max = [-15,15]
    N=100
    
    NN=100
    spectrum01=np.zeros((NN,2))
    spectrum02=np.zeros((NN,2))
    energy0=np.zeros((NN,2))
    energy1=np.zeros((NN,2))
    energy2=np.zeros((NN,2))

    i=0
    for ng in [0,0.5]:
        for x0 in np.linspace(-2*np.pi,2*np.pi,NN/2):
            
            x = linspace(phi_min,phi_max,N)+x0
            args = {'k': E_L, 'x0': x0}
            
            U = U_ho(x, args);
            
    #        x_opt_min = optimize.fmin(U_ho, [0.0], (args,)
            
            Q_dim=3
            Q = np.diag([0,1,2]).astype(np.complex)
                    
            Q_ = np.zeros((Q_dim,Q_dim)).astype(np.complex)
            for m in range(0,Q_dim):
                for n in range(0,Q_dim):
                    Q_[m,n]=mod_kron(m+1,n)+mod_kron(m-1,n)
            
            K = np.kron (assemble_K(N,-4*E_CL,phi_min,phi_max) , Q)
            
            u = assemble_u_potential(N, U_ho, x, args)
            V = np.kron(assemble_V(N,u),Q)
            
            H = K + V
            
            u = -E_J * np.cos(x/2)
            V = np.kron(assemble_V(N,u),Q_)
            
            H = H +V 
            
    #        ng=0.5
            
            H += 4* E_C * np.kron(np.diag(ones(N)),np.diag((array([-1,0,1])-ng)**2))
            
            evals, evecs = solve_eigenproblem(H)
            
            spectrum01[i,:]=[x0,evals[1]-evals[0]]
            spectrum02[i,:]= [x0,evals[2]-evals[0]]
            energy0[i,:]= [x0,evals[0]]
            energy1[i,:]= [x0,evals[1]]
            energy2[i,:]= [x0,evals[2]]
            i+=1
    #        plt.plot(x0*np.ones(2), evals[1:3]-evals[0],'.-')
#        plt.figure()   
    plt.plot(spectrum01[:,0],spectrum01[:,1],'.')
    plt.plot(spectrum02[:,0],spectrum02[:,1],'.')
        
#        plt.figure()
#        plt.plot(energy0[:,0],energy0[:,1],'.-')
#        plt.plot(energy1[:,0],energy1[:,1],'.-')
#    plt.plot(spectrum02[:,0],spectrum02[:,1],'.-')

