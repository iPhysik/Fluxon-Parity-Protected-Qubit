
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
    E_L = 0.4
    E_CL = 5
    Delta_E_J =0.5
#    E_J=E_J-Delta_E_J
    
    phi_min,phi_max = [-15,15]
    N=20
    
    Q_dim=15
    Q_list = np.arange(Q_dim)
    NN=50

#    ng_list = np.linspace(np.min(Q_list),np.max(Q_list),NN)
    ng_list = np.linspace(0,6,NN)
    spectrum = np.zeros((NN,3))
    
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
    for x0 in [-pi]:
        H=np.zeros((Q_dim*N,Q_dim*N)).astype(np.complex)
        x = linspace(phi_min,phi_max,N)+x0
        args = {'k': E_L, 'x0': x0}
        U = U_ho(x, args);
        u = assemble_u_potential(N, U_ho, x, args)
        V = np.kron(Q,assemble_V(N,u))
        
        # kinetic energy of Superinductor
        K = np.kron (Q,assemble_K(N,-4*E_CL,x.min(),x.max()))
        H = K+V
        
        u = -E_J * np.cos(x/2)
        V = np.kron(Q_, assemble_V(N,u)) 
        H +=V
        
        # asymmetry in CPB
        if False:
            u = - 1./2 *Delta_E_J* np.sin(x/2)
            V = np.kron(Q_p,assemble_V(N,u)) 
            H += V 
            
        for ng in ng_list:
            
            # charging energy of CPB
            H0 = 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(ones(N)) )
            Hf = H+H0

            evals, evecs = solve_eigenproblem(Hf)
            evals = evals.real
            
            spectrum[i,0]=ng
            spectrum[i,1:]= evals[1:3]-evals[0]
            i+=1
#            plt.plot(evals[1:10]-evals[0],'.-')
            
    plt.plot(spectrum[:,0],spectrum[:,1],'o-')
#    plt.plot(spectrum[:,0],spectrum[:,2],'o-')
#    plt.plot(spectrum[:,0],spectrum[:,3],'o-')


