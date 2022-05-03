#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:34:45 2019

@author: semyon
"""

import numpy as np
import cupy as cp
#import imageio as imo 
from numpy.fft import rfftn, irfftn
from numpy.fft import fftshift, ifftshift
from numpy.fft import fftfreq, rfftfreq
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import raw_data
import os
import re

load_directory = "C:\\Users\\Efremov-PC\\Desktop\\StarShapedTesselations\\Monoclinic\\HomogenizationData"
save_directory = "C:\\Users\\Efremov-PC\\Desktop\\StarShapedTesselations\\Monoclinic\\ElasticityTensors\\"
save_folder_list = ['monoclinic', 'orthotropic', 'transversely isotropic', 'isotropic', 'anisotropic', 'minimal Poissons']

structure_dimensions = np.array([80, 80, 80])

Young_mod_max = 1000.0
Young_mod_min = 1.0                                                                                                                            #Young modulus maximum value
Poisson = 0.3                                                                                                                                   #Poisson ratio

mean_strain = np.array([1.0,0.0,0.0,0.0,0.0,0.0])
mean_stress = np.array([1.0,0.0,0.0,0.0,0.0,0.0])

method_precision = 0.001
iteration_number = 20                                                                                                                          #The method precision

Tx = structure_dimensions[0] / np.min(structure_dimensions)                                                                                                                             #Domain size
Ty = structure_dimensions[1] / np.min(structure_dimensions)
Tz = structure_dimensions[2] / np.min(structure_dimensions)














def DataGenerator(size):
    
    result = np.ones((size[0], size[1], size[2]))
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                a = (i > (size[0] / 5)) and (i < (size[0] - (size[0] / 5)))
                b = (j > (size[1] / 5)) and (j < (size[1] - (size[1] / 5)))
                c = (k > (size[2] / 5)) and (k < (size[2] - (size[2] / 5)))
                if ((a and b) or (b and c) or (c and a)):
                    result[i,j,k] = 0
                    
    return result

def data_reader(size, name):

    file = open(name)
    data_raw = file.read().split()
    data_raw = list(map(int, data_raw))
    Data = np.array(data_raw)
    Data = np.reshape(Data, (size[0], size[1], size[2]), order='C')

    return Data

def color_plot(DataSet,Title):
    
    for i in range(np.size(DataSet[:,0])):
        for j in range(np.size(DataSet[0,:])):
            
            if (DataSet[i,j] > 1000):
                
                DataSet[i,j] = 1000
            
    fig = plt.figure(figsize = (20,20))
    cc = plt.imshow(DataSet.T,extent=[0,1.0,0,1.0],cmap='Greys') #plt.imshow(DataSet,extent=[0,1,0,1],cmap='Spectral')
    cbar = plt.colorbar(cc)

    plt.title(Title, fontsize=20)
    
    plt.grid(False)

    plt.show()

def strain_stress_color_plot(strain, stress):
    
    gridsize = (2, 6)
    fig = plt.figure(figsize=(48, 18))
    ax11 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=1)
    ax12 = plt.subplot2grid(gridsize, (0, 1), colspan=1, rowspan=1)
    ax13 = plt.subplot2grid(gridsize, (0, 2), colspan=1, rowspan=1)
    ax14 = plt.subplot2grid(gridsize, (0, 3), colspan=1, rowspan=1)
    ax15 = plt.subplot2grid(gridsize, (0, 4), colspan=1, rowspan=1)
    ax16 = plt.subplot2grid(gridsize, (0, 5), colspan=1, rowspan=1)
    
    ax21 = plt.subplot2grid(gridsize, (1, 0), colspan=1, rowspan=1)
    ax22 = plt.subplot2grid(gridsize, (1, 1), colspan=1, rowspan=1)
    ax23 = plt.subplot2grid(gridsize, (1, 2), colspan=1, rowspan=1)
    ax24 = plt.subplot2grid(gridsize, (1, 3), colspan=1, rowspan=1)
    ax25 = plt.subplot2grid(gridsize, (1, 4), colspan=1, rowspan=1)
    ax26 = plt.subplot2grid(gridsize, (1, 5), colspan=1, rowspan=1)
    
    ax11.set_xlabel('x')
    ax11.set_ylabel('y')
    ax12.set_xlabel('x')
    ax12.set_ylabel('y')
    ax13.set_xlabel('x')
    ax13.set_ylabel('y')
    ax14.set_xlabel('x')
    ax14.set_ylabel('y')
    ax15.set_xlabel('x')
    ax15.set_ylabel('y')
    ax16.set_xlabel('x')
    ax16.set_ylabel('y')
    
    ax21.set_xlabel('x')
    ax21.set_ylabel('y')
    ax22.set_xlabel('x')
    ax22.set_ylabel('y')
    ax23.set_xlabel('x')
    ax23.set_ylabel('y')
    ax24.set_xlabel('x')
    ax24.set_ylabel('y')
    ax25.set_xlabel('x')
    ax25.set_ylabel('y')
    ax26.set_xlabel('x')
    ax26.set_ylabel('y')
    
    #ax1.set_title('amplitude-based normalization',fontsize=20)
    #ax2.set_title('spectral+variance normalization',fontsize=20)
    
    #ax1.grid(True)
    #ax2.grid(True)
    #ax3.grid(True)
    #ax4.grid(True)
    #ax5.grid(True)
    #ax6.grid(True)
    
    clrbr11 = ax11.imshow(strain[0,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr11, ax = ax11)
    clrbr12 = ax12.imshow(strain[1,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr12, ax = ax12)
    clrbr13 = ax13.imshow(strain[2,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr13, ax = ax13)
    clrbr14 = ax14.imshow(strain[3,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr14, ax = ax14)
    clrbr15 = ax15.imshow(strain[4,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr15, ax = ax15)
    clrbr16 = ax16.imshow(strain[5,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr16, ax = ax16)
    
    clrbr21 = ax21.imshow(stress[0,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr21, ax = ax21)
    clrbr22 = ax22.imshow(stress[1,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr22, ax = ax22)
    clrbr23 = ax23.imshow(stress[2,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr23, ax = ax23)
    clrbr24 = ax24.imshow(stress[3,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr24, ax = ax24)
    clrbr25 = ax25.imshow(stress[4,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr25, ax = ax25)
    clrbr26 = ax26.imshow(stress[5,:,:].T,extent=[0,Tx,0,Ty],cmap='coolwarm')
    fig.colorbar(clrbr26, ax = ax26)
     
    plt.show()

def data_loader(Data, You_max, You_min, Poisson_const, x_length, y_length, z_length, size):
    
    n_x = size[0]
    n_y = size[1]
    n_z = size[2]
    
    Y_distr = You_min*np.ones((n_x, n_y, n_z))
    
    for i in range(n_x):
        for j in range(n_y):
            for v in range(n_z):
                
                if Data[i,j,v] == 1.0:
                    
                    Y_distr[i,j,v] = You_max
                    
                else:
                    
                    Y_distr[i,j,v] = You_min
    
    Y_distr_gpu = cp.asarray(Y_distr)
    
    lame_1 = np.ones((n_x, n_y, n_z))
    lame_2 = np.ones((n_x, n_y, n_z))
    
    lame_1_gpu = cp.asarray(lame_1)
    lame_1_gpu = (Poisson_const / ((1.0 + Poisson_const)*(1.0 - 2.0*Poisson_const)))*Y_distr_gpu
    lame_1 = cp.asnumpy(lame_1_gpu)
    del lame_1_gpu
    
    lame_2_gpu = cp.asarray(lame_2)
    lame_2_gpu = (0.5 / (1.0 + Poisson_const))*Y_distr_gpu
    lame_2 = cp.asnumpy(lame_2_gpu)
    del lame_2_gpu, Y_distr_gpu
    
    x_lattice = np.linspace(0.0, x_length, n_x + 1)
    y_lattice = np.linspace(0.0, y_length, n_y + 1)
    z_lattice = np.linspace(0.0, z_length, n_z + 1)
        
    return Y_distr, lame_1, lame_2, x_lattice[0:n_x], y_lattice[0:n_y], z_lattice[0:n_z], n_x, n_y, n_z

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def frequency_distribution(delta_x, delta_y, delta_z, n_x, n_y, n_z):
    
    frequency_distribution_x = fftshift(fftfreq(n_x, delta_x))
    frequency_distribution_y = fftshift(fftfreq(n_y, delta_y))
    frequency_distribution_z = fftshift(fftfreq(n_z, delta_z))
    
    return frequency_distribution_x, frequency_distribution_y, frequency_distribution_z

def tensor_product_gpu(a, b):
    
    result = cp.zeros((cp.size(a[:,0,0,0,0]), cp.size(b[0,:,0,0,0]), cp.size(a[0,0,:,0,0]), 6, 6))
    
    result[:,:,:,0,0] = a[:,:,:,0,0]*b[:,:,:,0,0]
    result[:,:,:,0,1] = a[:,:,:,0,0]*b[:,:,:,1,1]
    result[:,:,:,0,2] = a[:,:,:,0,0]*b[:,:,:,2,2]
    result[:,:,:,1,0] = a[:,:,:,1,1]*b[:,:,:,0,0]
    result[:,:,:,1,1] = a[:,:,:,1,1]*b[:,:,:,1,1]
    result[:,:,:,1,2] = a[:,:,:,1,1]*b[:,:,:,2,2]
    result[:,:,:,2,0] = a[:,:,:,2,2]*b[:,:,:,0,0]
    result[:,:,:,2,1] = a[:,:,:,2,2]*b[:,:,:,1,1]
    result[:,:,:,2,2] = a[:,:,:,2,2]*b[:,:,:,2,2]
    
    result[:,:,:,0,3] = a[:,:,:,0,0]*b[:,:,:,1,2]
    result[:,:,:,0,4] = a[:,:,:,0,0]*b[:,:,:,0,2]
    result[:,:,:,0,5] = a[:,:,:,0,0]*b[:,:,:,0,1]
    result[:,:,:,1,3] = a[:,:,:,1,1]*b[:,:,:,1,2]
    result[:,:,:,1,4] = a[:,:,:,1,1]*b[:,:,:,0,2]
    result[:,:,:,1,5] = a[:,:,:,1,1]*b[:,:,:,0,1]
    result[:,:,:,2,3] = a[:,:,:,2,2]*b[:,:,:,1,2]
    result[:,:,:,2,4] = a[:,:,:,2,2]*b[:,:,:,0,2]
    result[:,:,:,2,5] = a[:,:,:,2,2]*b[:,:,:,0,1]
    
    result[:,:,:,3,0] = a[:,:,:,1,2]*b[:,:,:,0,0]
    result[:,:,:,3,1] = a[:,:,:,1,2]*b[:,:,:,1,1]
    result[:,:,:,3,2] = a[:,:,:,1,2]*b[:,:,:,2,2]
    result[:,:,:,4,0] = a[:,:,:,0,2]*b[:,:,:,0,0]
    result[:,:,:,4,1] = a[:,:,:,0,2]*b[:,:,:,1,1]
    result[:,:,:,4,2] = a[:,:,:,0,2]*b[:,:,:,2,2]
    result[:,:,:,5,0] = a[:,:,:,0,1]*b[:,:,:,0,0]
    result[:,:,:,5,1] = a[:,:,:,0,1]*b[:,:,:,1,1]
    result[:,:,:,5,2] = a[:,:,:,0,1]*b[:,:,:,2,2]
    
    result[:,:,:,3,3] = a[:,:,:,1,2]*b[:,:,:,1,2]
    result[:,:,:,3,4] = a[:,:,:,1,2]*b[:,:,:,0,2]
    result[:,:,:,3,5] = a[:,:,:,1,2]*b[:,:,:,0,1]
    result[:,:,:,4,3] = a[:,:,:,0,2]*b[:,:,:,1,2]
    result[:,:,:,4,4] = a[:,:,:,0,2]*b[:,:,:,0,2]
    result[:,:,:,4,5] = a[:,:,:,0,2]*b[:,:,:,0,1]
    result[:,:,:,5,3] = a[:,:,:,0,1]*b[:,:,:,1,2]
    result[:,:,:,5,4] = a[:,:,:,0,1]*b[:,:,:,0,2]
    result[:,:,:,5,5] = a[:,:,:,0,1]*b[:,:,:,0,1]
    
    return result

def antisym_tensor_product_gpu(a, b):
    
    result = cp.zeros((cp.size(a[:,0,0,0,0]), cp.size(b[0,:,0,0,0]), cp.size(a[0,0,:,0,0]), 6, 6))
    
    result[:,:,:,0,0] = a[:,:,:,0,0]*b[:,:,:,0,0]
    result[:,:,:,0,1] = a[:,:,:,0,1]*b[:,:,:,0,1]
    result[:,:,:,0,2] = a[:,:,:,0,2]*b[:,:,:,0,2]
    result[:,:,:,1,0] = a[:,:,:,1,0]*b[:,:,:,1,0]
    result[:,:,:,1,1] = a[:,:,:,1,1]*b[:,:,:,1,1]
    result[:,:,:,1,2] = a[:,:,:,1,2]*b[:,:,:,1,2]
    result[:,:,:,2,0] = a[:,:,:,2,0]*b[:,:,:,2,0]
    result[:,:,:,2,1] = a[:,:,:,2,1]*b[:,:,:,2,1]
    result[:,:,:,2,2] = a[:,:,:,2,2]*b[:,:,:,2,2]
    
    result[:,:,:,0,3] = 0.5*(a[:,:,:,0,1]*b[:,:,:,0,2] + a[:,:,:,0,2]*b[:,:,:,0,1])
    result[:,:,:,0,4] = 0.5*(a[:,:,:,0,0]*b[:,:,:,0,2] + a[:,:,:,0,2]*b[:,:,:,0,0])
    result[:,:,:,0,5] = 0.5*(a[:,:,:,0,0]*b[:,:,:,0,1] + a[:,:,:,0,1]*b[:,:,:,0,0])
    result[:,:,:,1,3] = 0.5*(a[:,:,:,1,1]*b[:,:,:,1,2] + a[:,:,:,1,2]*b[:,:,:,1,1])
    result[:,:,:,1,4] = 0.5*(a[:,:,:,1,0]*b[:,:,:,1,2] + a[:,:,:,1,2]*b[:,:,:,1,0])
    result[:,:,:,1,5] = 0.5*(a[:,:,:,1,0]*b[:,:,:,1,1] + a[:,:,:,1,1]*b[:,:,:,1,0])
    result[:,:,:,2,3] = 0.5*(a[:,:,:,2,1]*b[:,:,:,2,2] + a[:,:,:,2,2]*b[:,:,:,2,1])
    result[:,:,:,2,4] = 0.5*(a[:,:,:,2,0]*b[:,:,:,2,2] + a[:,:,:,2,2]*b[:,:,:,2,0])
    result[:,:,:,2,5] = 0.5*(a[:,:,:,2,0]*b[:,:,:,2,1] + a[:,:,:,2,1]*b[:,:,:,2,0])
    
    result[:,:,:,3,0] = a[:,:,:,1,0]*b[:,:,:,2,0]
    result[:,:,:,3,1] = a[:,:,:,1,1]*b[:,:,:,2,1]
    result[:,:,:,3,2] = a[:,:,:,1,2]*b[:,:,:,2,2]
    result[:,:,:,4,0] = a[:,:,:,0,0]*b[:,:,:,2,0]
    result[:,:,:,4,1] = a[:,:,:,0,1]*b[:,:,:,2,1]
    result[:,:,:,4,2] = a[:,:,:,0,2]*b[:,:,:,2,2]
    result[:,:,:,5,0] = a[:,:,:,0,0]*b[:,:,:,1,0]
    result[:,:,:,5,1] = a[:,:,:,0,1]*b[:,:,:,1,1]
    result[:,:,:,5,2] = a[:,:,:,0,2]*b[:,:,:,1,2]
    
    result[:,:,:,3,3] = 0.5*(a[:,:,:,1,1]*b[:,:,:,2,2] + a[:,:,:,1,2]*b[:,:,:,2,1])
    result[:,:,:,3,4] = 0.5*(a[:,:,:,1,0]*b[:,:,:,2,2] + a[:,:,:,1,2]*b[:,:,:,2,0])
    result[:,:,:,3,5] = 0.5*(a[:,:,:,1,0]*b[:,:,:,2,1] + a[:,:,:,1,1]*b[:,:,:,2,0])
    result[:,:,:,4,3] = 0.5*(a[:,:,:,0,1]*b[:,:,:,2,2] + a[:,:,:,0,2]*b[:,:,:,2,1])
    result[:,:,:,4,4] = 0.5*(a[:,:,:,0,0]*b[:,:,:,2,2] + a[:,:,:,0,2]*b[:,:,:,2,0])
    result[:,:,:,4,5] = 0.5*(a[:,:,:,0,0]*b[:,:,:,2,1] + a[:,:,:,0,1]*b[:,:,:,2,0])
    result[:,:,:,5,3] = 0.5*(a[:,:,:,0,1]*b[:,:,:,1,2] + a[:,:,:,0,2]*b[:,:,:,1,1])
    result[:,:,:,5,4] = 0.5*(a[:,:,:,0,0]*b[:,:,:,1,2] + a[:,:,:,0,2]*b[:,:,:,1,0])
    result[:,:,:,5,5] = 0.5*(a[:,:,:,0,0]*b[:,:,:,1,1] + a[:,:,:,0,1]*b[:,:,:,1,0])
    
    return result

def basis_tensors_computation(frequency_x, frequency_y, frequency_z, lambda_0, mu_0):
    
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    freq_x_gpu = cp.asarray(frequency_x)
    freq_y_gpu = cp.asarray(frequency_y)
    freq_z_gpu = cp.asarray(frequency_z)
    k = cp.zeros((cp.size(freq_x_gpu), cp.size(freq_y_gpu), cp.size(freq_z_gpu), 3, 3))
    unit_2tensor = cp.zeros((cp.size(freq_x_gpu), cp.size(freq_y_gpu), cp.size(freq_z_gpu), 3, 3))
    unit_2tensor[:,:,:,0,0] = cp.ones((cp.size(freq_x_gpu), cp.size(freq_y_gpu), cp.size(freq_z_gpu)))
    unit_2tensor[:,:,:,1,1] = cp.ones((cp.size(freq_x_gpu), cp.size(freq_y_gpu), cp.size(freq_z_gpu)))
    unit_2tensor[:,:,:,2,2] = cp.ones((cp.size(freq_x_gpu), cp.size(freq_y_gpu), cp.size(freq_z_gpu)))
    common_freq = cp.zeros((cp.size(freq_x_gpu), cp.size(freq_y_gpu), cp.size(freq_z_gpu), 3))
    modulus = cp.zeros((cp.size(freq_x_gpu), cp.size(freq_y_gpu), cp.size(freq_z_gpu)))
    
    temp_freq = cp.meshgrid(freq_y_gpu, freq_x_gpu, freq_z_gpu)
    common_freq[:,:,:,0] = temp_freq[1]
    common_freq[:,:,:,1] = temp_freq[0]
    common_freq[:,:,:,2] = temp_freq[2]
    modulus = common_freq[:,:,:,0]**2.0 + common_freq[:,:,:,1]**2.0 + common_freq[:,:,:,2]**2.0
    modulus[cp.size(freq_x_gpu) // 2,cp.size(freq_y_gpu) // 2, cp.size(freq_z_gpu) // 2] = 1.0
    
    k[:,:,:,0,0] = (common_freq[:,:,:,0]**2.0) / modulus
    k[:,:,:,0,1] = (common_freq[:,:,:,0]*common_freq[:,:,:,1]) / modulus
    k[:,:,:,0,2] = (common_freq[:,:,:,0]*common_freq[:,:,:,2]) / modulus
    k[:,:,:,1,0] = (common_freq[:,:,:,1]*common_freq[:,:,:,0]) / modulus
    k[:,:,:,1,1] = (common_freq[:,:,:,1]**2.0) / modulus
    k[:,:,:,1,2] = (common_freq[:,:,:,1]*common_freq[:,:,:,2]) / modulus
    k[:,:,:,2,0] = (common_freq[:,:,:,2]*common_freq[:,:,:,0]) / modulus
    k[:,:,:,2,1] = (common_freq[:,:,:,2]*common_freq[:,:,:,1]) / modulus
    k[:,:,:,2,2] = (common_freq[:,:,:,2]**2.0) / modulus
    
    k_conj = unit_2tensor - k
    k_conj[cp.size(freq_x_gpu) // 2,cp.size(freq_y_gpu) // 2, cp.size(freq_z_gpu) // 2,:,:] = 0.0
    
    del unit_2tensor
    del freq_x_gpu
    del freq_y_gpu
    del common_freq
    del modulus
    del temp_freq
    
    k_cpu = cp.asnumpy(k)
    k_conj_cpu = cp.asnumpy(k_conj)
    
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    k = cp.asarray(k_cpu)
    k_conj = cp.asarray(k_conj_cpu)
    
    E_1 = 0.5*(tensor_product_gpu(k_conj, k_conj))
    E_1_cpu = cp.asnumpy(E_1)
    del E_1
    
    E_2 = tensor_product_gpu(k, k)
    E_2_cpu = cp.asnumpy(E_2)
    del E_2
    
    E_3 = antisym_tensor_product_gpu(k_conj, k_conj)
    E_3_cpu = cp.asnumpy(E_3)
    del E_3
    
    E_4 = antisym_tensor_product_gpu(k_conj, k) + antisym_tensor_product_gpu(k, k_conj)
    E_4_cpu = cp.asnumpy(E_4)
    del E_4
    
    del k
    del k_conj
    
    E_1 = cp.asarray(E_1_cpu)
    E_3 = cp.asarray(E_3_cpu)
    
    E_3 = cp.copy(E_3 - E_1)
    
    E_1_cpu = cp.asnumpy(E_1)
    E_3_cpu = cp.asnumpy(E_3)
    
    del E_1
    del E_3
    
    E_2 = cp.asarray(E_2_cpu)
    E_4 = cp.asarray(E_4_cpu)
    
    projector_strain = E_2 + E_4
    
    del E_2
    del E_4
    
    projector_strain = cp.swapaxes(projector_strain, 0, 2)
    projector_strain = cp.swapaxes(projector_strain, 0, 1)
    projector_strain = cp.swapaxes(projector_strain, 0, 3)
    projector_strain = cp.swapaxes(projector_strain, 1, 4)
    projector_strain[:,3,:,:,:] = 2.0*projector_strain[:,3,:,:,:]
    projector_strain[:,4,:,:,:] = 2.0*projector_strain[:,4,:,:,:]
    projector_strain[:,5,:,:,:] = 2.0*projector_strain[:,5,:,:,:]
    
    projector_strain_cpu = cp.asnumpy(projector_strain)
    
    del projector_strain
    
    E_2 = cp.asarray(E_2_cpu)
    E_4 = cp.asarray(E_4_cpu)
    
    k1 = 1.0 / (lambda_0 + 2.0*mu_0)
    k2 = 1.0 / (2.0*mu_0)
    green_tensor_strain = k1*E_2 + k2*E_4
    
    del E_2
    del E_4
    
    green_tensor_strain = cp.swapaxes(green_tensor_strain, 0, 2)
    green_tensor_strain = cp.swapaxes(green_tensor_strain, 0, 1)
    green_tensor_strain = cp.swapaxes(green_tensor_strain, 0, 3)
    green_tensor_strain = cp.swapaxes(green_tensor_strain, 1, 4)
    green_tensor_strain[:,3,:,:,:] = 2.0*green_tensor_strain[:,3,:,:,:]
    green_tensor_strain[:,4,:,:,:] = 2.0*green_tensor_strain[:,4,:,:,:]
    green_tensor_strain[:,5,:,:,:] = 2.0*green_tensor_strain[:,5,:,:,:]
    
    green_tensor_strain_cpu = cp.asnumpy(green_tensor_strain)
    
    del green_tensor_strain
    
    E_1 = cp.asarray(E_1_cpu)
    E_3 = cp.asarray(E_3_cpu)
    
    projector_stress = E_1 + E_3
    
    del E_1
    del E_3
    
    projector_stress = cp.swapaxes(projector_stress, 0, 2)
    projector_stress = cp.swapaxes(projector_stress, 0, 1)
    projector_stress = cp.swapaxes(projector_stress, 0, 3)
    projector_stress = cp.swapaxes(projector_stress, 1, 4)
    projector_stress[:,3,:,:,:] = 2.0*projector_stress[:,3,:,:,:]
    projector_stress[:,4,:,:,:] = 2.0*projector_stress[:,4,:,:,:]
    projector_stress[:,5,:,:,:] = 2.0*projector_stress[:,5,:,:,:]
    
    projector_stress_cpu = cp.asnumpy(projector_stress)
    
    del projector_stress
    
    E_1 = cp.asarray(E_1_cpu)
    E_3 = cp.asarray(E_3_cpu)
    
    k3 = 2.0*mu_0*(3.0*lambda_0 + 2.0*mu_0) / (lambda_0 + 2.0*mu_0)
    k4 = 2.0*mu_0
    green_tensor_stress = k3*E_1 + k4*E_3
    
    del E_1
    del E_3
    
    green_tensor_stress = cp.swapaxes(green_tensor_stress, 0, 2)
    green_tensor_stress = cp.swapaxes(green_tensor_stress, 0, 1)
    green_tensor_stress = cp.swapaxes(green_tensor_stress, 0, 3)
    green_tensor_stress = cp.swapaxes(green_tensor_stress, 1, 4)
    green_tensor_stress[:,3,:,:,:] = 2.0*green_tensor_stress[:,3,:,:,:]
    green_tensor_stress[:,4,:,:,:] = 2.0*green_tensor_stress[:,4,:,:,:]
    green_tensor_stress[:,5,:,:,:] = 2.0*green_tensor_stress[:,5,:,:,:]
    
    
    
    green_tensor_stress_cpu = cp.asnumpy(green_tensor_stress)
    #print(green_tensor_stress.nbytes)
    del green_tensor_stress
    
    print("GPU memry used in computations = ", mempool.used_bytes())
    print(mempool.total_bytes())
    
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    print("GPU memry used in computations = ", mempool.used_bytes())
    print(mempool.total_bytes())
    
    return projector_strain_cpu, projector_stress_cpu, green_tensor_strain_cpu, green_tensor_stress_cpu

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def double_dot_prod_stiffness(lame_1, lame_2, tensor):
    
    result = np.ones((np.shape(tensor)))
    result_gpu = cp.asarray(result)
    lame_1_gpu = cp.asarray(lame_1)
    lame_2_gpu = cp.asarray(lame_2)
    tensor_gpu = cp.asarray(tensor)
    
    result_gpu[0,:,:,:] = (lame_1_gpu[:,:,:] + 2.0*lame_2_gpu[:,:,:])*tensor_gpu[0,:,:,:] + lame_1_gpu[:,:,:]*tensor_gpu[1,:,:,:] + lame_1_gpu[:,:,:]*tensor_gpu[2,:,:,:]
    result_gpu[1,:,:,:] = lame_1_gpu[:,:,:]*tensor_gpu[0,:,:,:] + (lame_1_gpu[:,:,:] + 2.0*lame_2_gpu[:,:,:])*tensor_gpu[1,:,:,:] + lame_1_gpu[:,:,:]*tensor_gpu[2,:,:,:]
    result_gpu[2,:,:,:] = lame_1_gpu[:,:,:]*tensor_gpu[0,:,:,:] + lame_1_gpu[:,:,:]*tensor_gpu[1,:,:,:] + (lame_1_gpu[:,:,:] + 2.0*lame_2_gpu[:,:,:])*tensor_gpu[2,:,:,:]
    result_gpu[3,:,:,:] = 2.0*lame_2_gpu[:,:,:]*tensor_gpu[3,:,:,:]
    result_gpu[4,:,:,:] = 2.0*lame_2_gpu[:,:,:]*tensor_gpu[4,:,:,:]
    result_gpu[5,:,:,:] = 2.0*lame_2_gpu[:,:,:]*tensor_gpu[5,:,:,:]
    
    del lame_1_gpu, lame_2_gpu, tensor_gpu
    
    result = cp.asnumpy(result_gpu)
    
    return result

def double_dot_prod_gpu(A, b):
    
    result = np.zeros(np.shape(b), dtype = b.dtype)
    result_gpu = cp.asarray(result)
    A_gpu = cp.asarray(A)
    b_gpu = cp.asarray(b)
    
    result_gpu[0,:,:] = cp.sum(A_gpu[0,:,:,:]*b_gpu[:,:,:], axis=0)
    result_gpu[1,:,:] = cp.sum(A_gpu[1,:,:,:]*b_gpu[:,:,:], axis=0)
    result_gpu[2,:,:] = cp.sum(A_gpu[2,:,:,:]*b_gpu[:,:,:], axis=0)
    result_gpu[3,:,:] = cp.sum(A_gpu[3,:,:,:]*b_gpu[:,:,:], axis=0)
    result_gpu[4,:,:] = cp.sum(A_gpu[4,:,:,:]*b_gpu[:,:,:], axis=0)
    result_gpu[5,:,:] = cp.sum(A_gpu[5,:,:,:]*b_gpu[:,:,:], axis=0)
    
    result = cp.asnumpy(result_gpu)
            
    return result

def frobenius_norm_gpu(A):
    
    result = 0.0
    
    A_gpu = cp.asarray(A)
    
    result_gpu = cp.sqrt(cp.sum(cp.abs(A_gpu)**2.0))
    
    result = cp.asnumpy(result_gpu)
        
    return result

def convergence_test(projector, variable):
    
    #projection = np.zeros((3, np.size(stress[0,:,0]), np.size(stress[0,0,:])), dtype = np.complex128)
            
    projection = double_dot_prod_gpu(projector, variable)
    
    numerator = frobenius_norm_gpu(projection)
    denominator = frobenius_norm_gpu(variable)
    
    return numerator / denominator

def strain_lame_prescribed(lame_1_distribution, lame_2_distribution):
    
    return (1.0 / 2.0)*(np.min(lame_1_distribution) + np.max(lame_1_distribution)), (1.0 / 2.0)*(np.min(lame_2_distribution) + np.max(lame_2_distribution))

def initial_condition_strain_based(lame_1_distribution, lame_2_distribution, m_strain, n_x, n_y, n_z):
    
    strain_distribution = np.ones((6, n_x, n_y, n_z))
    stress_distribution = np.zeros((6, n_x, n_y, n_z))
    
    strain_distribution[0,:,:,:] = m_strain[0]*strain_distribution[0,:,:,:]
    strain_distribution[1,:,:,:] = m_strain[1]*strain_distribution[1,:,:,:]
    strain_distribution[2,:,:,:] = m_strain[2]*strain_distribution[2,:,:,:]
    strain_distribution[3,:,:,:] = m_strain[3]*strain_distribution[3,:,:,:]
    strain_distribution[4,:,:,:] = m_strain[4]*strain_distribution[4,:,:,:]
    strain_distribution[5,:,:,:] = m_strain[5]*strain_distribution[5,:,:,:]
    stress_distribution = double_dot_prod_stiffness(lame_1_distribution, lame_2_distribution, strain_distribution)
            
    return strain_distribution, stress_distribution

def iteration_strain_based(strain, stress, lame_1_distribution, lame_2_distribution, green_tensor, projector_tensor, frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, m_strain, lame_1_average, lame_2_average, n_x, n_y, n_z, epsilon):
    
    stress_temp = np.zeros((6, n_x, n_y, n_z))
    strain_temp = np.zeros((6, n_x, n_y, n_z))
    
    strain_gpu = cp.asarray(strain)
    
    strain_fourier_gpu = cp.fft.fftshift(cp.fft.fftn(strain_gpu, axes = (-3,-2,-1)), axes = (-3,-2,-1))
    
    del strain_gpu
    
    strain_fourier = cp.asnumpy(strain_fourier_gpu)
    
    del strain_fourier_gpu
    
    stress_gpu = cp.asarray(stress)
    
    stress_fourier_gpu = cp.fft.fftshift(cp.fft.fftn(stress_gpu, axes = (-3,-2,-1)), axes = (-3,-2,-1))
    
    del stress_gpu
    
    stress_fourier = cp.asnumpy(stress_fourier_gpu)
    
    del stress_fourier_gpu
    
    epsilon = convergence_test(projector_tensor, stress_fourier)
    
    doubdot = cp.asarray(double_dot_prod_gpu(green_tensor.real, stress_fourier))
    
    strain_fourier_gpu = cp.asarray(strain_fourier)
    
    strain_fourier_gpu = strain_fourier_gpu - doubdot
    strain_fourier_gpu[:,n_x // 2,n_y // 2,n_z // 2] = cp.asarray(n_x*n_y*n_z*m_strain)
    
    del doubdot
    
    #strain_fourier_gpu[0,n_x // 2,:] = cp.asarray(n_x*n_y*mean_strain[0]*np.ones(np.shape(strain_fourier[0,0,:])))
    #strain_fourier_gpu[1,n_x // 2,:] = cp.asarray(n_x*n_y*mean_strain[1]*np.ones(np.shape(strain_fourier[0,0,:])))
    #strain_fourier_gpu[2,n_x // 2,:] = cp.asarray(n_x*n_y*mean_strain[2]*np.ones(np.shape(strain_fourier[0,0,:])))
    #strain_fourier_gpu[0,:,n_y // 2] = cp.asarray(n_x*n_y*mean_strain[0]*np.ones(np.shape(strain_fourier[0,:,0])))
    #strain_fourier_gpu[1,:,n_y // 2] = cp.asarray(n_x*n_y*mean_strain[1]*np.ones(np.shape(strain_fourier[0,:,0])))
    #strain_fourier_gpu[2,:,n_y // 2] = cp.asarray(n_x*n_y*mean_strain[2]*np.ones(np.shape(strain_fourier[0,:,0])))
        
    #strain_fourier_gpu = cp.asarray(strain_fourier)
    
    strain_temp_1_gpu = cp.fft.ifftn(cp.fft.ifftshift(strain_fourier_gpu, axes = (-3,-2,-1)), axes = (-3,-2,-1))
    
    del strain_fourier_gpu
    
    strain_temp_1 = cp.asnumpy(strain_temp_1_gpu)
    
    del strain_temp_1_gpu
    
    strain_temp = strain_temp_1.real
    stress_temp = double_dot_prod_stiffness(lame_1_distribution, lame_2_distribution, strain_temp)
            
    return strain_temp, stress_temp, epsilon

def method_strain_based(lame_1_distribution, lame_2_distribution, m_strain, x_step, y_step, z_step, n_x, n_y, n_z):
    
    strain, stress = initial_condition_strain_based(lame_1_distribution, lame_2_distribution, m_strain, n_x, n_y, n_z)
    
    frequency_distribution_x, frequency_distribution_y, frequency_distribution_z = frequency_distribution(x_step, y_step, z_step, n_x, n_y, n_z)
    
    print(frequency_distribution_x[n_x // 2], frequency_distribution_y[n_y // 2], frequency_distribution_z[n_z // 2])
    
    lame_1_average, lame_2_average = strain_lame_prescribed(lame_1_distribution, lame_2_distribution)
    
    #green_tensor = green_tensor_composition(frequency_distribution_x, frequency_distribution_y, np.size(frequency_distribution_x), np.size(frequency_distribution_y), lame_1_average, lame_2_average)
    #projector_tensor = projector_tensor_composition(frequency_distribution_x, frequency_distribution_y, np.size(frequency_distribution_x), np.size(frequency_distribution_y))
    
    print("GPU memry used before = ", cp.get_default_memory_pool().used_bytes())
    
    projector_tensor, strain_projector_tensor, green_tensor, stress_green_tensor = basis_tensors_computation(frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, lame_1_average, lame_2_average)
    
    print("GPU memry used after = ", cp.get_default_memory_pool().used_bytes())
    
    #check_data_from_method(strain, m_strain, green_tensor, frequency_distribution_x, frequency_distribution_y, lame_1_average, lame_2_average, x_step, y_step, n_x, n_y)
    
    print('initial strain tensor = (',np.mean(strain[0,:,:,:]), np.mean(strain[1,:,:,:]), np.mean(strain[2,:,:,:]), np.mean(strain[3,:,:,:]), np.mean(strain[4,:,:,:]), np.mean(strain[5,:,:,:]),')')
    print('initial stress tensor = (',np.mean(stress[0,:,:,:]), np.mean(stress[1,:,:,:]), np.mean(stress[2,:,:,:]), np.mean(stress[3,:,:,:]), np.mean(stress[4,:,:,:]), np.mean(stress[5,:,:,:]),')')
    
    strain_stress_color_plot(strain[:,:,:,int(0.5 * n_z)], stress[:,:,:,int(0.5 * n_z)])
    
    epsilon = 1.0
    l = 0
    
    while (epsilon > method_precision):
        
        strain, stress, epsilon = iteration_strain_based(strain, stress, lame_1_distribution, lame_2_distribution, green_tensor, projector_tensor, frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, m_strain, lame_1_average, lame_2_average, n_x, n_y, n_z, epsilon)
        
        print('relative error (iteration no.', l+1, ') = ', epsilon)
        print('mean strain tensor (iteration no.', l+1, ') = (',np.mean(strain[0,:,:,:]), np.mean(strain[1,:,:,:]), np.mean(strain[2,:,:,:]), np.mean(strain[3,:,:,:]), np.mean(strain[4,:,:,:]), np.mean(strain[5,:,:,:]),')')
        print('mean stress tensor (iteration no.', l+1, ') = (',np.mean(stress[0,:,:,:]), np.mean(stress[1,:,:,:]), np.mean(stress[2,:,:,:]), np.mean(stress[3,:,:,:]), np.mean(stress[4,:,:,:]), np.mean(stress[5,:,:,:]),')')
        
        l+=1


    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('max strain tensor = (',np.max(strain[0,:,:,:]), np.max(strain[1,:,:,:]), np.max(strain[2,:,:,:]), np.max(strain[3,:,:,:]), np.max(strain[4,:,:,:]), np.max(strain[5,:,:,:]),')')
    print('min strain tensor = (',np.min(strain[0,:,:,:]), np.min(strain[1,:,:,:]), np.min(strain[2,:,:,:]), np.min(strain[3,:,:,:]), np.min(strain[4,:,:,:]), np.min(strain[5,:,:,:]),')')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')        
    print('max stress tensor = (',np.max(stress[0,:,:,:]), np.max(stress[1,:,:,:]), np.max(stress[2,:,:,:]), np.max(stress[3,:,:,:]), np.max(stress[4,:,:,:]), np.max(stress[5,:,:,:]),')')
    print('min stress tensor = (',np.min(stress[0,:,:,:]), np.min(stress[1,:,:,:]), np.min(stress[2,:,:,:]), np.min(stress[3,:,:,:]), np.min(stress[4,:,:,:]), np.min(stress[5,:,:,:]),')')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')        
    
    strain_stress_color_plot(strain[:,:,:,int(0.5 * n_z)], stress[:,:,:,int(0.5 * n_z)])
    
    mean_stress[0] = np.mean(stress[0,:,:,:])
    mean_stress[1] = np.mean(stress[1,:,:,:])
    mean_stress[2] = np.mean(stress[2,:,:,:])
    mean_stress[3] = np.mean(stress[3,:,:,:])
    mean_stress[4] = np.mean(stress[4,:,:,:])
    mean_stress[5] = np.mean(stress[5,:,:,:])
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')        
    print('average stress tensor = (', mean_stress,')')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    '''
    raw_data.write_ITK_metaimage(stress[0,:,:,:].astype(np.float32).T, 'stress00')
    raw_data.write_ITK_metaimage(stress[1,:,:,:].astype(np.float32).T, 'stress11')
    raw_data.write_ITK_metaimage(stress[2,:,:,:].astype(np.float32).T, 'stress22')
    raw_data.write_ITK_metaimage(stress[3,:,:,:].astype(np.float32).T, 'stress01')
    raw_data.write_ITK_metaimage(stress[4,:,:,:].astype(np.float32).T, 'stress02')
    raw_data.write_ITK_metaimage(stress[5,:,:,:].astype(np.float32).T, 'stress12')
    
    raw_data.write_ITK_metaimage(strain[0,:,:,:].astype(np.float32).T, 'strain00')
    raw_data.write_ITK_metaimage(strain[1,:,:,:].astype(np.float32).T, 'strain11')
    raw_data.write_ITK_metaimage(strain[2,:,:,:].astype(np.float32).T, 'strain22')
    raw_data.write_ITK_metaimage(strain[3,:,:,:].astype(np.float32).T, 'strain01')
    raw_data.write_ITK_metaimage(strain[4,:,:,:].astype(np.float32).T, 'strain02')
    raw_data.write_ITK_metaimage(strain[5,:,:,:].astype(np.float32).T, 'strain12')
    '''
    #projection_test(strain, stress, strain_projector_tensor, projector_tensor, n_x, n_y)
    
    return strain, stress, mean_stress






'''
#-------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def compliance_components_calculation(young_distr, poisson_ratio):
    
    component_1 = np.ones((np.shape(young_distr)))
    component_2 = np.ones((np.shape(young_distr)))
    
    component_1_gpu = cp.asarray(component_1)
    component_2_gpu = cp.asarray(component_2)
    young_distr_gpu = cp.asarray(young_distr)
    
    component_1_gpu = 1.0 / young_distr_gpu
    component_2_gpu = - cp.copy(poisson_ratio*component_1_gpu)
    
    component_1 = cp.asnumpy(component_1_gpu)
    component_2 = cp.asnumpy(component_2_gpu)
    
    del component_1_gpu, component_2_gpu, young_distr_gpu
    
    return component_1, component_2

def double_dot_prod_compliance(compliance_component_1, compliance_component_2, tensor):
    
    result = np.ones((np.shape(tensor)))
    result_gpu = cp.asarray(result)
    compliance_component_1_gpu = cp.asarray(compliance_component_1)
    compliance_component_2_gpu = cp.asarray(compliance_component_2)
    tensor_gpu = cp.asarray(tensor)
    
    result_gpu[0,:,:,:] = compliance_component_1_gpu[:,:,:]*tensor_gpu[0,:,:,:] + compliance_component_2_gpu[:,:,:]*tensor_gpu[1,:,:,:] + compliance_component_2_gpu[:,:,:]*tensor_gpu[2,:,:,:]
    result_gpu[1,:,:,:] = compliance_component_2_gpu[:,:,:]*tensor_gpu[0,:,:,:] + compliance_component_1_gpu[:,:,:]*tensor_gpu[1,:,:,:] + compliance_component_2_gpu[:,:,:]*tensor_gpu[2,:,:,:]
    result_gpu[2,:,:,:] = compliance_component_2_gpu[:,:,:]*tensor_gpu[0,:,:,:] + compliance_component_2_gpu[:,:,:]*tensor_gpu[1,:,:,:] + compliance_component_1_gpu[:,:,:]*tensor_gpu[2,:,:,:]
    result_gpu[3,:,:,:] = (compliance_component_1_gpu[:,:,:] - compliance_component_2_gpu[:,:,:])*tensor_gpu[3,:,:,:]
    result_gpu[4,:,:,:] = (compliance_component_1_gpu[:,:,:] - compliance_component_2_gpu[:,:,:])*tensor_gpu[4,:,:,:]
    result_gpu[5,:,:,:] = (compliance_component_1_gpu[:,:,:] - compliance_component_2_gpu[:,:,:])*tensor_gpu[5,:,:,:]
    
    del compliance_component_1_gpu, compliance_component_2_gpu, tensor_gpu
    
    result = cp.asnumpy(result_gpu)
    
    return result

def stress_lame_prescribed(lame_1_distribution, lame_2_distribution):
    
    return np.min(lame_1_distribution), np.min(lame_2_distribution)

def initial_condition_stress_based(compliance_1, compliance_2, m_stress, n_x, n_y, n_z):
    
    strain_distribution = np.zeros((6, n_x, n_y, n_z))
    stress_distribution = np.ones((6, n_x, n_y, n_z))
    
    stress_distribution[0,:,:,:] = m_stress[0]*stress_distribution[0,:,:,:]
    stress_distribution[1,:,:,:] = m_stress[1]*stress_distribution[1,:,:,:]
    stress_distribution[2,:,:,:] = m_stress[2]*stress_distribution[2,:,:,:]
    stress_distribution[3,:,:,:] = m_stress[3]*stress_distribution[3,:,:,:]
    stress_distribution[4,:,:,:] = m_stress[4]*stress_distribution[4,:,:,:]
    stress_distribution[5,:,:,:] = m_stress[5]*stress_distribution[5,:,:,:]
    strain_distribution = double_dot_prod_compliance(compliance_1, compliance_2, stress_distribution)
            
    return strain_distribution, stress_distribution

def iteration_stress_based(strain, stress, compliance_1, compliance_2, green_tensor, projector_tensor, frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, m_stress, lame_1_average, lame_2_average, n_x, n_y, n_z, epsilon):
    
    stress_temp = np.zeros((6, n_x, n_y, n_z))
    strain_temp = np.zeros((6, n_x, n_y, n_z))
    
    strain_gpu = cp.asarray(strain)
    
    strain_fourier_gpu = cp.fft.fftshift(cp.fft.fftn(strain_gpu, axes = (-3,-2,-1)), axes = (-3,-2,-1))
    
    del strain_gpu
    
    strain_fourier = cp.asnumpy(strain_fourier_gpu)
    
    del strain_fourier_gpu
    
    stress_gpu = cp.asarray(stress)
    
    stress_fourier_gpu = cp.fft.fftshift(cp.fft.fftn(stress_gpu, axes = (-3,-2,-1)), axes = (-3,-2,-1))
    
    del stress_gpu
    
    stress_fourier = cp.asnumpy(stress_fourier_gpu)
    
    del stress_fourier_gpu
    
    epsilon = convergence_test(projector_tensor, strain_fourier)
    
    doubdot = cp.asarray(double_dot_prod_gpu(green_tensor.real, strain_fourier))
    
    stress_fourier_gpu = cp.asarray(stress_fourier)
    
    stress_fourier_gpu = stress_fourier_gpu - doubdot
    stress_fourier_gpu[:,n_x // 2,n_y // 2,n_z // 2] = cp.asarray(n_x*n_y*n_z*m_stress)
    
    del doubdot
    
    #strain_fourier_gpu[0,n_x // 2,:] = cp.asarray(n_x*n_y*mean_strain[0]*np.ones(np.shape(strain_fourier[0,0,:])))
    #strain_fourier_gpu[1,n_x // 2,:] = cp.asarray(n_x*n_y*mean_strain[1]*np.ones(np.shape(strain_fourier[0,0,:])))
    #strain_fourier_gpu[2,n_x // 2,:] = cp.asarray(n_x*n_y*mean_strain[2]*np.ones(np.shape(strain_fourier[0,0,:])))
    #strain_fourier_gpu[0,:,n_y // 2] = cp.asarray(n_x*n_y*mean_strain[0]*np.ones(np.shape(strain_fourier[0,:,0])))
    #strain_fourier_gpu[1,:,n_y // 2] = cp.asarray(n_x*n_y*mean_strain[1]*np.ones(np.shape(strain_fourier[0,:,0])))
    #strain_fourier_gpu[2,:,n_y // 2] = cp.asarray(n_x*n_y*mean_strain[2]*np.ones(np.shape(strain_fourier[0,:,0])))
        
    #strain_fourier_gpu = cp.asarray(strain_fourier)
    
    stress_temp_1_gpu = cp.fft.ifftn(cp.fft.ifftshift(stress_fourier_gpu, axes = (-3,-2,-1)), axes = (-3,-2,-1))
    
    del stress_fourier_gpu
    
    stress_temp_1 = cp.asnumpy(stress_temp_1_gpu)
    
    del stress_temp_1_gpu
    
    stress_temp = stress_temp_1.real
    strain_temp = double_dot_prod_stiffness(compliance_1, compliance_2, stress_temp)
            
    return strain_temp, stress_temp, epsilon

def method_stress_based(compliance_1, compliance_2, lame_1_distribution, lame_2_distribution, m_stress, x_step, y_step, z_step, n_x, n_y, n_z):
    
    strain, stress = initial_condition_stress_based(compliance_1, compliance_2, m_stress, n_x, n_y, n_z)
    
    frequency_distribution_x, frequency_distribution_y, frequency_distribution_z = frequency_distribution(x_step, y_step, z_step, n_x, n_y, n_z)
    
    print(frequency_distribution_x[n_x // 2], frequency_distribution_y[n_y // 2], frequency_distribution_z[n_z // 2])
    
    lame_1_average, lame_2_average = stress_lame_prescribed(lame_1_distribution, lame_2_distribution)
    
    #green_tensor = green_tensor_composition(frequency_distribution_x, frequency_distribution_y, np.size(frequency_distribution_x), np.size(frequency_distribution_y), lame_1_average, lame_2_average)
    #projector_tensor = projector_tensor_composition(frequency_distribution_x, frequency_distribution_y, np.size(frequency_distribution_x), np.size(frequency_distribution_y))
    
    print("GPU memry used before = ", cp.get_default_memory_pool().used_bytes())
    
    projector_tensor, strain_projector_tensor, green_tensor, stress_green_tensor = basis_tensors_computation(frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, lame_1_average, lame_2_average)
    
    print("GPU memry used after = ", cp.get_default_memory_pool().used_bytes())
    
    #check_data_from_method(strain, m_strain, green_tensor, frequency_distribution_x, frequency_distribution_y, lame_1_average, lame_2_average, x_step, y_step, n_x, n_y)
    
    print('initial strain tensor = (',np.mean(strain[0,:,:,:]), np.mean(strain[1,:,:,:]), np.mean(strain[2,:,:,:]), np.mean(strain[3,:,:,:]), np.mean(strain[4,:,:,:]), np.mean(strain[5,:,:,:]),')')
    print('initial stress tensor = (',np.mean(stress[0,:,:,:]), np.mean(stress[1,:,:,:]), np.mean(stress[2,:,:,:]), np.mean(stress[3,:,:,:]), np.mean(stress[4,:,:,:]), np.mean(stress[5,:,:,:]),')')
    
    strain_stress_color_plot(strain[:,:,:,0], stress[:,:,:,0])
    
    epsilon = 1.0
    l = 0
    
    while (epsilon > method_precision):
        
        strain, stress, epsilon = iteration_stress_based(strain, stress, compliance_1, compliance_2, stress_green_tensor, strain_projector_tensor, frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, m_stress, lame_1_average, lame_2_average, n_x, n_y, n_z, epsilon)
        
        print('relative error (iteration no.', l+1, ') = ', epsilon)
        print('mean strain tensor (iteration no.', l+1, ') = (',np.mean(strain[0,:,:,:]), np.mean(strain[1,:,:,:]), np.mean(strain[2,:,:,:]), np.mean(strain[3,:,:,:]), np.mean(strain[4,:,:,:]), np.mean(strain[5,:,:,:]),')')
        print('mean stress tensor (iteration no.', l+1, ') = (',np.mean(stress[0,:,:,:]), np.mean(stress[1,:,:,:]), np.mean(stress[2,:,:,:]), np.mean(strain[3,:,:,:]), np.mean(strain[4,:,:,:]), np.mean(strain[5,:,:,:]),')')
        
        l+=1


    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('max strain tensor = (',np.max(strain[0,:,:,:]), np.max(strain[1,:,:,:]), np.max(strain[2,:,:,:]), np.max(strain[3,:,:,:]), np.max(strain[4,:,:,:]), np.max(strain[5,:,:,:]),')')
    print('min strain tensor = (',np.min(strain[0,:,:,:]), np.min(strain[1,:,:,:]), np.min(strain[2,:,:,:]), np.min(strain[3,:,:,:]), np.min(strain[4,:,:,:]), np.min(strain[5,:,:,:]),')')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')        
    print('max stress tensor = (',np.max(stress[0,:,:,:]), np.max(stress[1,:,:,:]), np.max(stress[2,:,:,:]), np.max(stress[3,:,:,:]), np.max(stress[4,:,:,:]), np.max(stress[5,:,:,:]),')')
    print('min stress tensor = (',np.min(stress[0,:,:,:]), np.min(stress[1,:,:,:]), np.min(stress[2,:,:,:]), np.min(stress[3,:,:,:]), np.min(stress[4,:,:,:]), np.min(stress[5,:,:,:]),')')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')        
    
    strain_stress_color_plot(strain[:,:,:,0], stress[:,:,:,0])
    '''
    mean_stress[0] = np.mean(stress[0,:,:])
    mean_stress[1] = np.mean(stress[1,:,:])
    mean_stress[2] = np.mean(stress[2,:,:])
    
    #projection_test(strain, stress, strain_projector_tensor, projector_tensor, n_x, n_y)
    '''
    return projector_tensor, strain_projector_tensor, green_tensor, stress_green_tensor

'''
#-------------------------------------------------------------------------------------------------------------------------------------------------------
'''

def MaterialType(compliance):
    
    norm = 0.0
    
    for i in range(6):
        norm = norm + compliance[i,i]
    
    norm = norm / 6.0
    

def FileSetReader(folder):
    
    file_name_list = os.listdir(folder)
    closed_structure_name_list = [f for f in file_name_list if re.match("cl*", f)]
    open_structure_name_list = [f for f in file_name_list if re.match("op*", f)]
    
    return(closed_structure_name_list, open_structure_name_list)

def Exploration(filename_list_closed, filename_list_open, load_folder, save_folder, size):
    
    
    for name in filename_list_open:
            
        print('processing the structure ', name)
        
        structure = np.loadtxt((load_folder + "\\" + name))
        name = name[:-8]
        structure = np.reshape(structure, (size[0], size[1], size[2]))
        
        mean_stiffness = np.zeros((6,6))
        mean_compliance = np.zeros((6,6))
        Y_distr, lame_1, lame_2, x_lattice, y_lattice, z_lattice, n_x, n_y, n_z = data_loader(structure, Young_mod_max, Young_mod_min, Poisson, Tx, Ty, Tz)
        compliance_1, compliance_2 = compliance_components_calculation(Y_distr, Poisson)
        x_step = x_lattice[1] - x_lattice[0]
        y_step = y_lattice[1] - y_lattice[0]
        z_step = z_lattice[1] - z_lattice[0]
        
        #color_plot(Y_distr[:,:,0], 'Young Modulus')
        
        for iter_direction in range(6):
            
            mean_strain = np.zeros(6)
            mean_strain[iter_direction] = 1.0
            strain, stress, mean_stress = method_strain_based(lame_1, lame_2, mean_strain, x_step, y_step, z_step, n_x, n_y, n_z)
            mean_stiffness[iter_direction,:] = np.copy(mean_stress)
            if (iter_direction > 2):
                mean_stiffness[iter_direction,:] = 0.5 * mean_stiffness[iter_direction,:]
            
        mean_stiffness = np.transpose(mean_stiffness)
        #for i in range(6):
        #    for j in range(6):
        #        if (np.abs(mean_stiffness[i,j]) <= (method_precision)):
        #            mean_stiffness[i,j] = 0.0
        mean_compliance = np.linalg.inv(mean_stiffness)
        
        np.savetxt(save_folder + name + "_stiff.txt", mean_stiffness)
        np.savetxt(save_folder + name + "_compl.txt", mean_compliance)
        raw_data.write_ITK_metaimage(stress[0,:,:,:].astype(np.float32).T, save_folder + name + "_s00")
        raw_data.write_ITK_metaimage(stress[1,:,:,:].astype(np.float32).T, save_folder + name + "_s11")
        raw_data.write_ITK_metaimage(stress[2,:,:,:].astype(np.float32).T, save_folder + name + "_s22")
        raw_data.write_ITK_metaimage(stress[3,:,:,:].astype(np.float32).T, save_folder + name + "_s12")
        raw_data.write_ITK_metaimage(stress[4,:,:,:].astype(np.float32).T, save_folder + name + "_s02")
        raw_data.write_ITK_metaimage(stress[5,:,:,:].astype(np.float32).T, save_folder + name + "_s01")
        raw_data.write_ITK_metaimage(strain[0,:,:,:].astype(np.float32).T, save_folder + name + "_e00")
        raw_data.write_ITK_metaimage(strain[1,:,:,:].astype(np.float32).T, save_folder + name + "_e11")
        raw_data.write_ITK_metaimage(strain[2,:,:,:].astype(np.float32).T, save_folder + name + "_e22")
        raw_data.write_ITK_metaimage(strain[3,:,:,:].astype(np.float32).T, save_folder + name + "_e12")
        raw_data.write_ITK_metaimage(strain[4,:,:,:].astype(np.float32).T, save_folder + name + "_e02")
        raw_data.write_ITK_metaimage(strain[5,:,:,:].astype(np.float32).T, save_folder + name + "_e01")
        
    for name in filename_list_closed:
        
        print('processing the structure ', name)
        
        structure = np.loadtxt((load_folder + "\\" + name))
        name = name[:-8]
        structure = np.reshape(structure, (size[0], size[1], size[2]))
        
        mean_stiffness = np.zeros((6,6))
        mean_compliance = np.zeros((6,6))
        Y_distr, lame_1, lame_2, x_lattice, y_lattice, z_lattice, n_x, n_y, n_z = data_loader(structure, Young_mod_max, Young_mod_min, Poisson, Tx, Ty, Tz)
        compliance_1, compliance_2 = compliance_components_calculation(Y_distr, Poisson)
        x_step = x_lattice[1] - x_lattice[0]
        y_step = y_lattice[1] - y_lattice[0]
        z_step = z_lattice[1] - z_lattice[0]
        
        #color_plot(Y_distr[:,:,0], 'Young Modulus')
        
        for iter_direction in range(6):
            
            mean_strain = np.zeros(6)
            mean_strain[iter_direction] = 1.0
            strain, stress, mean_stress = method_strain_based(lame_1, lame_2, mean_strain, x_step, y_step, z_step, n_x, n_y, n_z)
            mean_stiffness[iter_direction,:] = np.copy(mean_stress)
            if (iter_direction > 2):
                mean_stiffness[iter_direction,:] = 0.5 * mean_stiffness[iter_direction,:]
            
        mean_stiffness = np.transpose(mean_stiffness)
        #for i in range(6):
        #    for j in range(6):
        #        if (np.abs(mean_stiffness[i,j]) <= (method_precision)):
        #            mean_stiffness[i,j] = 0.0
        mean_compliance = np.linalg.inv(mean_stiffness)
        
        np.savetxt(save_folder + name + '_stiff.txt', mean_stiffness)
        np.savetxt(save_folder + name + '_compl.txt', mean_compliance)
        raw_data.write_ITK_metaimage(stress[0,:,:,:].astype(np.float32).T, save_folder + name + '_s00')
        raw_data.write_ITK_metaimage(stress[1,:,:,:].astype(np.float32).T, save_folder + name + '_s11')
        raw_data.write_ITK_metaimage(stress[2,:,:,:].astype(np.float32).T, save_folder + name + '_s22')
        raw_data.write_ITK_metaimage(stress[3,:,:,:].astype(np.float32).T, save_folder + name + '_s12')
        raw_data.write_ITK_metaimage(stress[4,:,:,:].astype(np.float32).T, save_folder + name + '_s02')
        raw_data.write_ITK_metaimage(stress[5,:,:,:].astype(np.float32).T, save_folder + name + '_s01')
        raw_data.write_ITK_metaimage(strain[0,:,:,:].astype(np.float32).T, save_folder + name + '_e00')
        raw_data.write_ITK_metaimage(strain[1,:,:,:].astype(np.float32).T, save_folder + name + '_e11')
        raw_data.write_ITK_metaimage(strain[2,:,:,:].astype(np.float32).T, save_folder + name + '_e22')
        raw_data.write_ITK_metaimage(strain[3,:,:,:].astype(np.float32).T, save_folder + name + '_e12')
        raw_data.write_ITK_metaimage(strain[4,:,:,:].astype(np.float32).T, save_folder + name + '_e02')
        raw_data.write_ITK_metaimage(strain[5,:,:,:].astype(np.float32).T, save_folder + name + '_e01')
        

















name = "cl_0_0_8_3_3_14_0_0_3_hom.txt"

print("start")
structure = data_reader(structure_dimensions, load_directory + "\\" + name)
#structure = data_reader(structure_dimensions, "tetragonal.txt")
#structure = DataGenerator(structure_dimensions)
print("start")

#frequency_x, frequency_y, frequency_z = frequency_distribution(x_lattice[1] - x_lattice[0], y_lattice[1] - y_lattice[0], z_lattice[1] - z_lattice[0], 150, 150, 150)

#projector_strain_cpu, projector_stress_cpu, green_tensor_strain_cpu, green_tensor_stress_cpu = basis_tensors_computation(frequency_x, frequency_y, frequency_z, 0.35, 0.25)
#structure = np.reshape(structure, (structure_dimensions[0], structure_dimensions[1], structure_dimensions[2]))
        
mean_stiffness = np.zeros((6,6))
mean_compliance = np.zeros((6,6))
Y_distr, lame_1, lame_2, x_lattice, y_lattice, z_lattice, n_x, n_y, n_z = data_loader(structure, Young_mod_max, Young_mod_min, Poisson, Tx, Ty, Tz, structure_dimensions)
compliance_1, compliance_2 = compliance_components_calculation(Y_distr, Poisson)
x_step = x_lattice[1] - x_lattice[0]
y_step = y_lattice[1] - y_lattice[0]
z_step = z_lattice[1] - z_lattice[0]
        
#color_plot(Y_distr[:,:,0], 'Young Modulus')

for iter_direction in range(6):
    
    mean_strain = np.zeros(6)
    mean_strain[iter_direction] = 1.0
    strain, stress, mean_stress = method_strain_based(lame_1, lame_2, mean_strain, x_step, y_step, z_step, n_x, n_y, n_z)
    mean_stiffness[iter_direction,:] = np.copy(mean_stress)
    if (iter_direction > 2):
        mean_stiffness[iter_direction,:] = 0.5 * mean_stiffness[iter_direction,:]
    
mean_stiffness = np.transpose(mean_stiffness)
#for i in range(6):
#    for j in range(6):
#        if (np.abs(mean_stiffness[i,j]) <= (method_precision)):
#            mean_stiffness[i,j] = 0.0
mean_compliance = np.linalg.inv(mean_stiffness)

np.savetxt(save_directory + name, mean_stiffness)

#closed_structure_name_list, open_structure_name_list = FileSetReader(load_directory)
#Exploration(closed_structure_name_list, open_structure_name_list, load_directory, save_directory, structure_dimensions)