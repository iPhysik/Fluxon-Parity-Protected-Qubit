# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:01:53 2018

@author: Wenyuan Zhang @ Rutgers <wzhang@physics.rutgers.edu>
"""

import numpy as np
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


def spectrum_vs_ng(ng_list,Q_list,phi0,params,args,asymEJ=True):
    E_J =params['EJ']
    E_C = params['EC']
    E_L = params['EL']
    E_CL = params['ECL']
    Delta_E_J =params['Delta_EJ']
    
    phi_min = args['Phi_min']
    phi_max = args['Phi_max']
    gridsize=args['gridsize'] # grid size along coordinate phi
    
    Q = np.diag(Q_list).astype(np.complex)
    
    Q_dim = np.size(Q_list)
    Q_ = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_[m,n]=mod_kron(m+1,n)+mod_kron(m-1,n)
            
    Q_p = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_p[m,n]=-1j*mod_kron(m+1,n)+1j*mod_kron(m-1,n)
            
    
    # potential energy of superinductor
    x0=phi0
    x = linspace(phi_min,phi_max,gridsize+1)+x0
    u = assemble_u_potential(U_ho, x, {'k': E_L, 'x0': x0})
    V = np.kron(np.diag(ones(Q_dim)),assemble_V(u))
    
    # kinetic energy of superinductor
    K = np.kron (np.diag(ones(Q_dim)),assemble_K(-4*E_CL,x))
    
    #
    H = K + V
    
    # symmetric CPB
    u = -E_J * np.cos(x/2)
    V = np.kron(Q_, assemble_V(u)) 
    H +=V
    
    # add asymmetry to EJ
    if asymEJ:
        u = - 1./2 *Delta_E_J* np.sin(x/2)
        V = np.kron(Q_p,assemble_V(u)) 
        H += V 

    
    i=0
    spectrum = np.zeros((np.size(ng_list),3))
    for ng in ng_list:
        # charging energy of CPB
        H0 = 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(ones(gridsize+1)) )
        Hf = H+H0
    
        evals, evecs = solve_eigenproblem(Hf)
        evals = evals.real
    
        spectrum[i,0]=ng
        spectrum[i,1:]= evals[1:3]-evals[0]
        i+=1
        
    return spectrum


def spectrum_vs_phi0(phi0_list,Q_list,ng_list,params,args,asymEJ=True):
    E_J =params['EJ']
    E_C = params['EC']
    E_L = params['EL']
    E_CL = params['ECL']
    Delta_E_J =params['Delta_EJ']
    
    phi_min = args['Phi_min']
    phi_max = args['Phi_max']
    gridsize=args['gridsize'] # grid size along coordinate phi
    
    Q = np.diag(Q_list).astype(np.complex)
    
    Q_dim = np.size(Q_list)
    Q_ = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_[m,n]=mod_kron(m+1,n)+mod_kron(m-1,n)
            
    Q_p = np.zeros((Q_dim,Q_dim)).astype(np.complex)
    for m in range(0,Q_dim):
        for n in range(0,Q_dim):
            Q_p[m,n]=-1j*mod_kron(m+1,n)+1j*mod_kron(m-1,n)
            
    
    i=0
    spectrum = np.zeros((np.size(ng_list)*np.size(phi0_list),3))

    for ng in ng_list:
        # charging energy of CPB
        H0 = 4* E_C * np.kron( (np.diag((Q_list-ng)**2)), np.diag(ones(gridsize+1)) )
        
        for x0 in phi0_list:
            # potential energy of superinductor
            x = linspace(phi_min,phi_max,gridsize+1)+x0
            u = assemble_u_potential(U_ho, x, {'k': E_L, 'x0': x0})
            V = np.kron(np.diag(ones(Q_dim)),assemble_V(u))
            
            # kinetic energy of superinductor
            K = np.kron (np.diag(ones(Q_dim)),assemble_K(-4*E_CL,x))
            
            #
            H = K + V
            
            # symmetric CPB
            u = -E_J * np.cos(x/2)
            V = np.kron(Q_, assemble_V(u)) 
            H +=V
            
            # add asymmetry to EJ
            if asymEJ:
                u = - 1./2 *Delta_E_J* np.sin(x/2)
                V = np.kron(Q_p,assemble_V(u)) 
                H += V 
        
            evals, evecs = solve_eigenproblem(H+H0)
            evals = evals.real
        
            spectrum[i,0]=x0
            spectrum[i,1:]= evals[1:3]-evals[0]
            i+=1
            
    return spectrum

