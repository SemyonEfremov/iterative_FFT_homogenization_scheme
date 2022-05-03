# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:03:30 2021

@author: MSI PC
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

load_directory = "C:\\Users\\Efremov-PC\\Documents\\Visual Studio 2019\\star-shaped-dinstances-master\\star-shaped-dinstances-master\\3DCellGrowthStarShaped\\3DCellGrowthStarShaped\\metadata\\SymmetryTests\\homogenization"
save_directory = "C:\\Users\\Efremov-PC\\Documents\\Visual Studio 2019\\star-shaped-dinstances-master\\star-shaped-dinstances-master\\3DCellGrowthStarShaped\\3DCellGrowthStarShaped\\metadata\\SymmetryTests\\homogenization\\homogenized\\"
save_folder_list = ['monoclinic', 'orthotropic', 'transversely isotropic', 'isotropic', 'anisotropic', 'minimal Poissons']

structure_dimensions = np.array([50, 50, 150])

Young_mod_max = 100.0
Young_mod_min = 1.0                                                                                                             #Young modulus maximum value
Poisson = 0.3                                                                                                                                   #Poisson ratio

mean_strain = np.array([1.0,0.0,0.0,0.0,0.0,0.0])
mean_stress = np.array([1.0,0.0,0.0,0.0,0.0,0.0])

method_precision = 0.0001
iteration_number = 20                                                                                                                          #The method precision

Tx = structure_dimensions[0] / np.min(structure_dimensions)                                                                                                                             #Domain size
Ty = structure_dimensions[1] / np.min(structure_dimensions)
Tz = structure_dimensions[2] / np.min(structure_dimensions)







#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------

def DeformationDataToVTK(structure, displacements, strain_field, stress_field, size, step, name):
    
    v_file = open(name + ".vtk", "w")
    v_file.write("# vtk DataFile Version 3.0\n")
    v_file.write("deformation data\n")
    v_file.write("ASCII\n")
    v_file.write("DATASET STRUCTURED_POINTS\n")
    v_file.write("DIMENSIONS " + str(size[0]) + " " + str(size[1]) + " " + str(size[2]) + "\n")
    v_file.write("ORIGIN 0.0 0.0 0.0\n")
    v_file.write("SPACING " + str(step[0]) + " " + str(step[1]) + " " + str(step[2]) + "\n")
    
    v_file.write("POINT_DATA " + str(size[0] * size[1] * size[2]) + "\n")
    v_file.write("SCALARS structure unsigned_int\n")
    v_file.write("LOOKUP_TABLE default\n")
    for k in range(size[2]):
        for j in range(size[1]):
            for i in range(size[0]):
                
                v_file.write(str(structure[i, j, k]))
                v_file.write(" ")
                
    v_file.write("\n")
    
    v_file.write("VECTORS displacements float\n")
    for k in range(size[2]):
        for j in range(size[1]):
            for i in range(size[0]):
                
                v_file.write(str(displacements[0, i, j, k]) + " " + str(displacements[1, i, j, k]) + " " + str(displacements[2, i, j, k]))
                v_file.write("\n")
                
    v_file.write("TENSORS strain float\n")
    for k in range(size[2]):
        for j in range(size[1]):
            for i in range(size[0]):
                
                v_file.write(str(strain_field[0, i, j, k]) + " " + str(strain_field[5, i, j, k]) + " " + str(strain_field[4, i, j, k]))
                v_file.write("\n")
                v_file.write(str(strain_field[5, i, j, k]) + " " + str(strain_field[1, i, j, k]) + " " + str(strain_field[3, i, j, k]))
                v_file.write("\n")
                v_file.write(str(strain_field[4, i, j, k]) + " " + str(strain_field[3, i, j, k]) + " " + str(strain_field[2, i, j, k]))
                v_file.write("\n\n")
    
    v_file.write("TENSORS stress float\n")
    for k in range(size[2]):
        for j in range(size[1]):
            for i in range(size[0]):
                
                v_file.write(str(stress_field[0, i, j, k]) + " " + str(stress_field[5, i, j, k]) + " " + str(stress_field[4, i, j, k]))
                v_file.write("\n")
                v_file.write(str(stress_field[5, i, j, k]) + " " + str(stress_field[1, i, j, k]) + " " + str(stress_field[3, i, j, k]))
                v_file.write("\n")
                v_file.write(str(stress_field[4, i, j, k]) + " " + str(stress_field[3, i, j, k]) + " " + str(stress_field[2, i, j, k]))
                v_file.write("\n\n")
                
    v_file.close()

#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------

def DataReader(size, name):

    file = open(name)
    data_raw = file.read().split()
    data_raw = list(map(int, data_raw))
    Data = np.array(data_raw)
    Data = np.reshape(Data, (size, size, size))

    return Data

def ColorPlot(DataSet,Title):
    
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

def StrainStressColorPlot(strain, stress):
    
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

#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------

def DataGenerator(size):
    
    result = np.ones((size[0], size[1], size[2]))
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                a = (i > (size[0] / 15)) and (i < (size[0] - (size[0] / 15)))
                b = (j > (size[1] / 15)) and (j < (size[1] - (size[1] / 15)))
                c = (k > (size[2] / 15)) and (k < (size[2] - (size[2] / 15)))
                if ((a and b) or (b and c) or (c and a)):
                    result[i,j,k] = 0
                    
    return result

def DataLoader(Data, You_max, You_min, Poisson_const, x_length, y_length, z_length, size):
    
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
    
    
        
    return Y_distr, lame_1, lame_2

#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------

def GridComputation(number_of_voxels, x_length, y_length, z_length):
    
    x_grid = np.linspace(0.0, x_length, number_of_voxels[0] + 1)
    y_grid = np.linspace(0.0, y_length, number_of_voxels[1] + 1)
    z_grid = np.linspace(0.0, z_length, number_of_voxels[2] + 1)
    
    step_x = x_grid[1] - x_grid[0]
    step_y = y_grid[1] - y_grid[0]
    step_z = z_grid[1] - z_grid[0]
    
    return x_grid[0:number_of_voxels[0]], y_grid[0:number_of_voxels[1]], z_grid[0:number_of_voxels[2]], step_x, step_y, step_z

def FrequencyDistribution(n_x, n_y, n_z):
    
    frequency_distribution_x = 2.0 * np.pi * fftfreq(n_x)
    frequency_distribution_y = 2.0 * np.pi * fftfreq(n_y)
    frequency_distribution_z = 2.0 * np.pi * fftfreq(n_z)
    
    return frequency_distribution_x, frequency_distribution_y, frequency_distribution_z

def DiscreteFrequencyMultipier(frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, step_x, step_y, step_z):
    
    frequency_distribution_x_GPU = cp.asarray(frequency_distribution_x)
    multiplier_positive_x_GPU = (1.0 / step_x) * (cp.exp(1j * cp.copy(frequency_distribution_x_GPU)) - 1)
    multiplier_negative_x_GPU = - (1.0 / step_x) * (cp.exp(-1j * cp.copy(frequency_distribution_x_GPU)) - 1)
    multiplier_positive_x = cp.asnumpy(multiplier_positive_x_GPU)
    multiplier_negative_x = cp.asnumpy(multiplier_negative_x_GPU)
    
    del frequency_distribution_x_GPU, multiplier_positive_x_GPU, multiplier_negative_x_GPU
    
    frequency_distribution_y_GPU = cp.asarray(frequency_distribution_y)
    multiplier_positive_y_GPU = (1.0 / step_y) * (cp.exp(1j * cp.copy(frequency_distribution_y_GPU)) - 1)
    multiplier_negative_y_GPU = - (1.0 / step_y) * (cp.exp(-1j * cp.copy(frequency_distribution_y_GPU)) - 1)
    multiplier_positive_y = cp.asnumpy(multiplier_positive_y_GPU)
    multiplier_negative_y = cp.asnumpy(multiplier_negative_y_GPU)
    
    del frequency_distribution_y_GPU, multiplier_positive_y_GPU, multiplier_negative_y_GPU
    
    frequency_distribution_z_GPU = cp.asarray(frequency_distribution_z)
    multiplier_positive_z_GPU = (1.0 / step_z) * (cp.exp(1j * cp.copy(frequency_distribution_z_GPU)) - 1)
    multiplier_negative_z_GPU = - (1.0 / step_z) * (cp.exp(-1j * cp.copy(frequency_distribution_z_GPU)) - 1)
    multiplier_positive_z = cp.asnumpy(multiplier_positive_z_GPU)
    multiplier_negative_z = cp.asnumpy(multiplier_negative_z_GPU)
    
    del frequency_distribution_z_GPU, multiplier_positive_z_GPU, multiplier_negative_z_GPU
    
    return multiplier_positive_x, multiplier_negative_x, multiplier_positive_y, multiplier_negative_y, multiplier_positive_z, multiplier_negative_z

def DeltaLameDistribution(lame_1_distribution, lame_2_distribution, lame_1_reference_material, lame_2_reference_material):
    
    delta_lame_1_distribution_GPU = cp.asarray(lame_1_distribution) - lame_1_reference_material
    delta_lame_1_distribution = cp.asnumpy(delta_lame_1_distribution_GPU)
    del delta_lame_1_distribution_GPU
    delta_lame_2_distribution_GPU = cp.asarray(lame_2_distribution) - lame_2_reference_material
    delta_lame_2_distribution = cp.asnumpy(delta_lame_2_distribution_GPU)
    del delta_lame_2_distribution_GPU
    
    return delta_lame_1_distribution, delta_lame_2_distribution

def LameCoefficientsPrescribed(lame_1_distribution, lame_2_distribution):
    
    return (1.0 / 2.0)*(np.min(lame_1_distribution) + np.max(lame_1_distribution)), (1.0 / 2.0)*(np.min(lame_2_distribution) + np.max(lame_2_distribution))

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

def MeanTensorComputation(tensor_data):
    
    result = np.zeros(np.size(tensor_data[:,0,0,0]))
    result = np.copy(cp.asnumpy(cp.mean(cp.asarray(tensor_data), axis=(1,2,3))))
    
    return result

#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------

def ForwardFiniteDifferenceCPU(data, step, axis_id):
    
    result = np.zeros(np.shape(data))
    for i in range(np.size(data[0,:,0,0])):
        for j in range(np.size(data[0,0,:,0])):
            for k in range(np.size(data[0,0,0,:])):
                if (axis_id == 0):
                    if (i < (np.size(data[0,:,0,0]) - 1)):
                        result[:,i,j,k] = (1.0 / step) * (data[:,i + 1,j,k] - data[:,i,j,k])
                    else:
                        result[:,i,j,k] = (1.0 / step) * (data[:,0,j,k] - data[:,i,j,k])
                if (axis_id == 1):
                    if (j < (np.size(data[0,0,:,0]) - 1)):
                        result[:,i,j,k] = (1.0 / step) * (data[:,i,j + 1,k] - data[:,i,j,k])
                    else:
                        result[:,i,j,k] = (1.0 / step) * (data[:,i,0,k] - data[:,i,j,k])
                if (axis_id == 2):
                    if (k < (np.size(data[0,0,0,:]) - 1)):
                        result[:,i,j,k] = (1.0 / step) * (data[:,i,j,k + 1] - data[:,i,j,k])
                    else:
                        result[:,i,j,k] = (1.0 / step) * (data[:,i,j,0] - data[:,i,j,k])
                
    return result

def ForwardFiniteDifference(data, step, axis_id):
    
    data_GPU = cp.asarray(data)
    result = (1.0 / step) * cp.asnumpy(cp.roll(data_GPU, -1, axis=axis_id) - data_GPU)
    del data_GPU
    return result

def BackwardFiniteDifference(data, step, axis_id):
    
    data_GPU = cp.asarray(data)
    result = (1.0 / step) * cp.asnumpy(data_GPU - cp.roll(data_GPU, 1, axis=axis_id))
    del data_GPU
    return result

def TensorTestGenerator(x, y, z, size):
    
    result = np.zeros((6, size[0], size[1], size[2]))
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                result[3,i,j,k] = x[i] + 2 * y[j] + 4 * z[k]
                result[4,i,j,k] = x[i] + 2 * y[j] + 4 * z[k]
                result[5,i,j,k] = x[i] + 2 * y[j] + 4 * z[k]
                
    return result

def DivergenceTest(tensor_data, k_d):
    
    result = np.zeros((3, np.size(tensor_data[0,:,0,0]), np.size(tensor_data[0,0,:,0]), np.size(tensor_data[0,0,0,:])))
    result_fourier = np.zeros((3, np.size(tensor_data[0,:,0,0]), np.size(tensor_data[0,0,:,0]), np.size(tensor_data[0,0,0,:])), dtype=np.complex128)
    k_n = -np.conj(k_d)
    temp_tensor_fourier = np.fft.fftn(tensor_data, axes = (-3,-2,-1))
    result_fourier[0,:,:,:] = k_n[0,:,:,:] * temp_tensor_fourier[0,:,:,:] + k_d[1,:,:,:] * temp_tensor_fourier[5,:,:,:] + k_d[2,:,:,:] * temp_tensor_fourier[4,:,:,:]
    result_fourier[1,:,:,:] = k_d[0,:,:,:] * temp_tensor_fourier[5,:,:,:] + k_n[1,:,:,:] * temp_tensor_fourier[1,:,:,:] + k_d[2,:,:,:] * temp_tensor_fourier[3,:,:,:]
    result_fourier[2,:,:,:] = k_d[0,:,:,:] * temp_tensor_fourier[4,:,:,:] + k_d[1,:,:,:] * temp_tensor_fourier[3,:,:,:] + k_n[2,:,:,:] * temp_tensor_fourier[2,:,:,:]
    result = np.fft.ifftn(result_fourier, axes = (-3,-2,-1)).real
    
    return result

def Divergence(tensor_data, step_x, step_y, step_z):
    
    result = np.zeros((3, np.size(tensor_data[0,:,0,0]), np.size(tensor_data[0,0,:,0]), np.size(tensor_data[0,0,0,:])))
    result[0,:,:,:] = np.copy(cp.asnumpy(cp.asarray(BackwardFiniteDifference(tensor_data[0,:,:,:], step_x, 0)) + cp.asarray(ForwardFiniteDifference(tensor_data[5,:,:,:], step_y, 1)) + cp.asarray(ForwardFiniteDifference(tensor_data[4,:,:,:], step_z, 2))))
    result[1,:,:,:] = np.copy(cp.asnumpy(cp.asarray(ForwardFiniteDifference(tensor_data[5,:,:,:], step_x, 0)) + cp.asarray(BackwardFiniteDifference(tensor_data[1,:,:,:], step_y, 1)) + cp.asarray(ForwardFiniteDifference(tensor_data[3,:,:,:], step_z, 2))))
    result[2,:,:,:] = np.copy(cp.asnumpy(cp.asarray(ForwardFiniteDifference(tensor_data[4,:,:,:], step_x, 0)) + cp.asarray(ForwardFiniteDifference(tensor_data[3,:,:,:], step_y, 1)) + cp.asarray(BackwardFiniteDifference(tensor_data[2,:,:,:], step_z, 2))))
    
    return result

def StrainTensorComputation(displacement_field, mean_strain, step_x, step_y, step_z):
    
    result = np.zeros((6, np.size(displacement_field[0,:,0,0]), np.size(displacement_field[0,0,:,0]), np.size(displacement_field[0,0,0,:])))
    result[0,:,:,:] = ForwardFiniteDifference(displacement_field[0,:,:,:], step_x, 0)
    result[1,:,:,:] = ForwardFiniteDifference(displacement_field[1,:,:,:], step_y, 1)
    result[2,:,:,:] = ForwardFiniteDifference(displacement_field[2,:,:,:], step_z, 2)
    result[3,:,:,:] = cp.asnumpy(0.5 * (cp.asarray(BackwardFiniteDifference(displacement_field[1,:,:,:], step_z, 2)) + cp.asarray(BackwardFiniteDifference(displacement_field[2,:,:,:], step_y, 1))))
    result[4,:,:,:] = cp.asnumpy(0.5 * (cp.asarray(BackwardFiniteDifference(displacement_field[0,:,:,:], step_z, 2)) + cp.asarray(BackwardFiniteDifference(displacement_field[2,:,:,:], step_x, 0))))
    result[5,:,:,:] = cp.asnumpy(0.5 * (cp.asarray(BackwardFiniteDifference(displacement_field[0,:,:,:], step_y, 1)) + cp.asarray(BackwardFiniteDifference(displacement_field[1,:,:,:], step_x, 0))))
    
    mean_strain_GPU = cp.asarray(mean_strain)
    result[0,:,:,:] = cp.asnumpy(cp.asarray(result[0,:,:,:]) + mean_strain_GPU[0])
    result[1,:,:,:] = cp.asnumpy(cp.asarray(result[1,:,:,:]) + mean_strain_GPU[1])
    result[2,:,:,:] = cp.asnumpy(cp.asarray(result[2,:,:,:]) + mean_strain_GPU[2])
    result[3,:,:,:] = cp.asnumpy(cp.asarray(result[3,:,:,:]) + mean_strain_GPU[3])
    result[4,:,:,:] = cp.asnumpy(cp.asarray(result[4,:,:,:]) + mean_strain_GPU[4])
    result[5,:,:,:] = cp.asnumpy(cp.asarray(result[5,:,:,:]) + mean_strain_GPU[5])
    del mean_strain_GPU
    
    return result

def GreenFunctionPrecomputation(k_d, lambda_0, mu_0):
    
    result_temp = np.zeros((3, 3, np.size(k_d[0,:,0,0]), np.size(k_d[1,0,:,0]), np.size(k_d[2,0,0,:])), dtype=np.complex128)
    coef = (mu_0 + lambda_0) / (2.0 * mu_0 + lambda_0)
    k_c = cp.asnumpy(-cp.conj(cp.asarray(k_d)))
    k_mod = cp.asnumpy(cp.sum(cp.asarray(k_d) * (-cp.array(k_c)), axis=0).real)
    k_mod[0,0,0] = 1.0
    k_d_GPU = cp.asarray(k_d)
    k_c_GPU = cp.asarray(k_c)
    
    result_temp[0,0,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[0,:,:,:] * k_d_GPU[0,:,:,:]))
    result_temp[1,1,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[1,:,:,:] * k_d_GPU[1,:,:,:]))
    result_temp[2,2,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[2,:,:,:] * k_d_GPU[2,:,:,:]))
    result_temp[0,1,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[0,:,:,:] * k_d_GPU[1,:,:,:]))
    result_temp[0,2,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[0,:,:,:] * k_d_GPU[2,:,:,:]))
    result_temp[1,0,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[1,:,:,:] * k_d_GPU[0,:,:,:]))
    result_temp[1,2,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[1,:,:,:] * k_d_GPU[2,:,:,:]))
    result_temp[2,0,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[2,:,:,:] * k_d_GPU[0,:,:,:]))
    result_temp[2,1,:,:,:] = np.copy(cp.asnumpy(k_c_GPU[2,:,:,:] * k_d_GPU[1,:,:,:]))
    
    del k_d_GPU, k_c_GPU
    
    k_mod_GPU = 1.0 / (mu_0 * cp.asarray(k_mod))
    k_mod_GPU[0,0,0] = 0.0
    
    result_temp[0,0,:,:,:] = np.copy(cp.asnumpy(k_mod_GPU + (coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[0,0,:,:,:])))
    result_temp[1,1,:,:,:] = np.copy(cp.asnumpy(k_mod_GPU + (coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[1,1,:,:,:])))
    result_temp[2,2,:,:,:] = np.copy(cp.asnumpy(k_mod_GPU + (coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[2,2,:,:,:])))
    result_temp[0,1,:,:,:] = np.copy(cp.asnumpy((coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[0,1,:,:,:])))
    result_temp[0,2,:,:,:] = np.copy(cp.asnumpy((coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[0,2,:,:,:])))
    result_temp[1,0,:,:,:] = np.copy(cp.asnumpy((coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[1,0,:,:,:])))
    result_temp[1,2,:,:,:] = np.copy(cp.asnumpy((coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[1,2,:,:,:])))
    result_temp[2,0,:,:,:] = np.copy(cp.asnumpy((coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[2,0,:,:,:])))
    result_temp[2,1,:,:,:] = np.copy(cp.asnumpy((coef * mu_0 * k_mod_GPU * k_mod_GPU) * cp.asarray(result_temp[2,1,:,:,:])))
    
    result_temp[:,:,0,0,0] = 0 * result_temp[:,:,0,0,0]
    del k_mod_GPU
    
    return result_temp

def GreenFunctionMultiplication(u, green_function):
    
    result = np.zeros(np.shape(u), dtype=np.complex128)
    green_function_GPU = cp.asarray(green_function)
    u_GPU = cp.asarray(u)
    
    result[0,:,:,:] = cp.asnumpy(green_function_GPU[0,0,:,:,:] * u_GPU[0,:,:,:] + green_function_GPU[0,1,:,:,:] * u_GPU[1,:,:,:] + green_function_GPU[0,2,:,:,:] * u_GPU[2,:,:,:])
    result[1,:,:,:] = cp.asnumpy(green_function_GPU[1,0,:,:,:] * u_GPU[0,:,:,:] + green_function_GPU[1,1,:,:,:] * u_GPU[1,:,:,:] + green_function_GPU[1,2,:,:,:] * u_GPU[2,:,:,:])
    result[2,:,:,:] = cp.asnumpy(green_function_GPU[2,0,:,:,:] * u_GPU[0,:,:,:] + green_function_GPU[2,1,:,:,:] * u_GPU[1,:,:,:] + green_function_GPU[2,2,:,:,:] * u_GPU[2,:,:,:])
    del green_function_GPU, u_GPU
    
    return result

#u - displacement_field_fourier, k_d - a meshgrid of positive discrete frequency multipliers
def GreenFunctionOperator(u, k_d, lambda_0, mu_0):
    
    '''print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("GPU memory used at the beginning = ", cp.get_default_memory_pool().used_bytes())'''
    result = np.zeros((3, np.size(k_d[0,:,0,0]), np.size(k_d[1,0,:,0]), np.size(k_d[2,0,0,:])), dtype=np.complex128)
    coef = (mu_0 + lambda_0) / (mu_0 * (2.0 * mu_0 + lambda_0))
    k_c = cp.asnumpy(-cp.conj(cp.asarray(k_d)))
    k_mod = cp.asnumpy(cp.sum(cp.asarray(k_d) * (-cp.array(k_c)), axis=0).real)
    k_mod[0,0,0] = 1.0
    k_d_GPU = cp.asarray(k_d)
    k_mod_GPU = cp.asarray(k_mod)
    u_GPU = cp.asarray(u)
    k_c_GPU = cp.asarray(k_c[0,:,:,:])
    #print("GPU memory used meanwhile = ", cp.get_default_memory_pool().used_bytes())
    result[0,:,:,:] = np.copy(cp.asnumpy((1.0 / (mu_0 * k_mod_GPU)) * u_GPU[0,:,:,:] + (coef / (k_mod_GPU * k_mod_GPU)) * k_c_GPU * (k_d_GPU[0,:,:,:] * u_GPU[0,:,:,:] + k_d_GPU[1,:,:,:] * u_GPU[1,:,:,:] + k_d_GPU[2,:,:,:] * u_GPU[2,:,:,:])))
    k_c_GPU = cp.asarray(k_c[1,:,:,:])
    result[1,:,:,:] = np.copy(cp.asnumpy((1.0 / (mu_0 * k_mod_GPU)) * u_GPU[1,:,:,:] + (coef / (k_mod_GPU * k_mod_GPU)) * k_c_GPU * (k_d_GPU[0,:,:,:] * u_GPU[0,:,:,:] + k_d_GPU[1,:,:,:] * u_GPU[1,:,:,:] + k_d_GPU[2,:,:,:] * u_GPU[2,:,:,:])))
    k_c_GPU = cp.asarray(k_c[2,:,:,:])
    result[2,:,:,:] = np.copy(cp.asnumpy((1.0 / (mu_0 * k_mod_GPU)) * u_GPU[2,:,:,:] + (coef / (k_mod_GPU * k_mod_GPU)) * k_c_GPU * (k_d_GPU[0,:,:,:] * u_GPU[0,:,:,:] + k_d_GPU[1,:,:,:] * u_GPU[1,:,:,:] + k_d_GPU[2,:,:,:] * u_GPU[2,:,:,:])))
    result[:,0,0,0] = np.array([0.0,0.0,0.0])
    #print("GPU memory used meanwhile = ", cp.get_default_memory_pool().used_bytes())
    del u_GPU, k_d_GPU, k_c_GPU, k_mod_GPU
    '''print("GPU memory used in the end = ", cp.get_default_memory_pool().used_bytes())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")'''
    
    return result

def RelativeErrorComputationDivergence(stress, step_x, step_y, step_z):
    
    temp_divergence_GPU = cp.asarray(Divergence(stress, step_x, step_y, step_z))
    numerator = cp.asnumpy(cp.mean(temp_divergence_GPU[0,:,:,:]**2.0 + temp_divergence_GPU[1,:,:,:]**2.0 + temp_divergence_GPU[2,:,:,:]**2.0))
    del temp_divergence_GPU
    stress_GPU = cp.asarray(stress)
    denominator = cp.asnumpy(cp.mean(stress_GPU[0,:,:,:])**2.0 + cp.mean(stress_GPU[1,:,:,:])**2.0 + cp.mean(stress_GPU[2,:,:,:])**2.0)
    del stress_GPU
    
    return np.sqrt(numerator / denominator)

def RelativeErrorComputationWillot(stress, step_x, step_y, step_z):
    
    temp_divergence_GPU = cp.asarray(Divergence(stress, step_x, step_y, step_z))
    numerator = cp.asnumpy(cp.mean(temp_divergence_GPU[0,:,:,:]**2.0 + temp_divergence_GPU[1,:,:,:]**2.0 + temp_divergence_GPU[2,:,:,:]**2.0))
    del temp_divergence_GPU
    stress_GPU = cp.asarray(stress)
    denominator = cp.asnumpy(cp.mean(stress_GPU[0,:,:,:])**2.0 + cp.mean(stress_GPU[1,:,:,:])**2.0 + cp.mean(stress_GPU[2,:,:,:])**2.0)
    del stress_GPU
    
    return np.sqrt(numerator / denominator)

def RelativeErrorComputationImplicit(stress, stress_previous):
    
    numerator = cp.linalg.norm(cp.mean(cp.asarray(stress), axis=(-3,-2,-1)) - cp.mean(cp.asarray(stress_previous), axis=(-3,-2,-1)))
    denominator = cp.linalg.norm(cp.mean(cp.asarray(stress), axis=(-3,-2,-1)))
    
    return numerator / denominator

def RelativeErrorComputationMax(stress, stress_previous):
    
    numerator = cp.max(cp.linalg.norm((cp.asarray(stress) - cp.asarray(stress_previous)), axis = 0))
    denominator = cp.linalg.norm(cp.mean(cp.asarray(stress), axis=(-3,-2,-1)))
    
    return numerator / denominator

#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------

def WillotIteration(displacement_field, strain, lame_1_distribution, lame_2_distribution, delta_lame_1_distribution, delta_lame_2_distribution, green_function, mean_strain, k_d, lambda_0, mu_0, step_x, step_y, step_z, relative_error):
    
    displacement_field_temp = np.zeros(np.shape(displacement_field))
    strain_temp = np.zeros(np.shape(strain))
    arg_val = np.copy(double_dot_prod_stiffness(delta_lame_1_distribution, delta_lame_2_distribution, strain))
    displacement_field_temp = np.copy(Divergence(arg_val, step_x, step_y, step_z))
    del arg_val
    print("GPU memory used at the beginning = ", cp.get_default_memory_pool().used_bytes())
    displacement_field_GPU = cp.asarray(displacement_field_temp)
    print("GPU memory used at the beginning = ", cp.get_default_memory_pool().used_bytes())
    displacement_fourier_GPU = cp.fft.fftn(displacement_field_GPU, axes = (-3,-2,-1))
    
    del displacement_field_GPU
    
    displacement_fourier = cp.asnumpy(displacement_fourier_GPU)
    
    del displacement_fourier_GPU
    
    #displacement_fourier = GreenFunctionOperator(displacement_fourier, k_d, lambda_0, mu_0)
    displacement_fourier = GreenFunctionMultiplication(displacement_fourier, green_function)
    
    displacement_fourier_GPU = cp.asarray(displacement_fourier)
    
    displacement_field_GPU = cp.fft.ifftn(displacement_fourier_GPU, axes = (-3,-2,-1))
    
    del displacement_fourier_GPU
    
    displacement_field_temp = cp.asnumpy(displacement_field_GPU).real
    
    del displacement_field_GPU
    strain_temp = StrainTensorComputation(displacement_field_temp, mean_strain, step_x, step_y, step_z)
    stress_temp = double_dot_prod_stiffness(lame_1_distribution, lame_2_distribution, strain_temp)
    
    stress_previous = double_dot_prod_stiffness(lame_1_distribution, lame_2_distribution, strain)
    relative_error = RelativeErrorComputationMax(stress_temp, stress_previous)
    
    #relative_error = RelativeErrorComputationDivergence(stress_temp, step_x, step_y, step_z)
    
    return displacement_field_temp, strain_temp, stress_temp, relative_error

def WillotMethod(Data, You_max, You_min, Poisson_const, x_length, y_length, z_length, size):
    
    x_grid, y_grid, z_grid, step_x, step_y, step_z = GridComputation(size, x_length, y_length, z_length)
    frequency_distribution_x, frequency_distribution_y, frequency_distribution_z = FrequencyDistribution(0, 0, 0, size[0], size[1], size[2])
    k_p_x, k_n_x, k_p_y, k_n_y, k_p_z, k_n_z = DiscreteFrequencyMultipier(frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, step_x, step_y, step_z)
    k_d = np.zeros((3, size[0], size[1], size[2]))
    temp_k = cp.meshgrid(cp.asarray(k_p_y), cp.asarray(k_p_x), cp.asarray(k_p_z))
    k_d[0,:,:,:] = cp.asnumpy(temp_k[1])
    k_d[1,:,:,:] = cp.asnumpy(temp_k[0])
    k_d[2,:,:,:] = cp.asnumpy(temp_k[2])
    del temp_k
    young_mod_distribution, lame_1_distribution, lame_2_distribution = DataLoader(Data, You_max, You_min, Poisson_const, x_length, y_length, z_length, size)
    lame_1_reference_material, lame_2_reference_material = LameCoefficientsPrescribed(lame_1_distribution, lame_2_distribution)
    delta_lame_1_distribution, delta_lame_2_distribution = DeltaLameDistribution(lame_1_distribution, lame_2_distribution, lame_1_reference_material, lame_2_reference_material)
    
    displacement_field = np.zeros((np.shape(k_d)))
    strain = np.zeros((6, size[0], size[1], size[2]))
    stress = np.zeros((6, size[0], size[1], size[2]))
    strain[0,:,:,:] = strain[0,:,:,:] + mean_strain[0]
    strain[1,:,:,:] = strain[1,:,:,:] + mean_strain[1]
    strain[2,:,:,:] = strain[2,:,:,:] + mean_strain[2]
    relative_error = 1.0
    
    while (relative_error > method_precision):
        
        displacement_field, strain, relative_error = WillotIteration(displacement_field, strain, lame_1_distribution, lame_2_distribution, delta_lame_1_distribution, delta_lame_2_distribution, mean_strain, k_d, lame_1_reference_material, lame_2_reference_material, step_x, step_y, step_z, relative_error)
    
    return displacement_field, strain, stress

size = structure_dimensions
x_length = Tx
y_length = Ty
z_length = Tz
You_min = Young_mod_min
You_max = Young_mod_max
Poisson_const = Poisson
x_grid, y_grid, z_grid, step_x, step_y, step_z = GridComputation(size, x_length, y_length, z_length)
frequency_distribution_x, frequency_distribution_y, frequency_distribution_z = FrequencyDistribution(size[0], size[1], size[2])
k_p_x, k_n_x, k_p_y, k_n_y, k_p_z, k_n_z = DiscreteFrequencyMultipier(frequency_distribution_x, frequency_distribution_y, frequency_distribution_z, step_x, step_y, step_z)
k_d = np.zeros((3, size[0], size[1], size[2]), dtype=np.complex128)
temp_k = cp.meshgrid(cp.asarray(k_p_y), cp.asarray(k_p_x), cp.asarray(k_p_z))
k_d[0,:,:,:] = cp.asnumpy(temp_k[1])
k_d[1,:,:,:] = cp.asnumpy(temp_k[0])
k_d[2,:,:,:] = cp.asnumpy(temp_k[2])
del temp_k

#Data = DataGenerator(size)
Data = np.loadtxt(load_directory + "\\" + "op_-0.785398_-0.955317_8_1_1_7_0_1_6_hom.txt")
Data = np.reshape(Data, (size[0], size[1], size[2]), order='C')

young_mod_distribution, lame_1_distribution, lame_2_distribution = DataLoader(Data, You_max, You_min, Poisson_const, x_length, y_length, z_length, size)
lame_1_reference_material, lame_2_reference_material = LameCoefficientsPrescribed(lame_1_distribution, lame_2_distribution)
delta_lame_1_distribution, delta_lame_2_distribution = DeltaLameDistribution(lame_1_distribution, lame_2_distribution, lame_1_reference_material, lame_2_reference_material)

green_function = GreenFunctionPrecomputation(k_d, lame_1_reference_material, lame_2_reference_material)

mean_stiffness = np.zeros((6, 6))

for iter_direction in range(6):
            
    mean_strain = np.zeros(6)
    mean_strain[iter_direction] = 1.0
    displacement_field = np.zeros((np.shape(k_d)))
    strain = np.zeros((6, size[0], size[1], size[2]))
    stress = np.zeros((6, size[0], size[1], size[2]))
    strain[0,:,:,:] = strain[0,:,:,:] + mean_strain[0]
    strain[1,:,:,:] = strain[1,:,:,:] + mean_strain[1]
    strain[2,:,:,:] = strain[2,:,:,:] + mean_strain[2]
    relative_error = 1.0
    iteration_id = 0
    while (relative_error > method_precision):
        
        iteration_id += 1
        displacement_field, strain, stress, relative_error = WillotIteration(displacement_field, strain, lame_1_distribution, lame_2_distribution, delta_lame_1_distribution, delta_lame_2_distribution, green_function, mean_strain, k_d, lame_1_reference_material, lame_2_reference_material, step_x, step_y, step_z, relative_error)
        print("average displacement = ", np.mean(displacement_field, axis=(1,2,3)))
        print("average strain = ", np.mean(strain, axis=(1,2,3)))
        print("average stress = ", np.mean(stress, axis=(1,2,3)))
        print("iteration no. ", iteration_id, ">>>>>>>>> ", relative_error, " <<<<<<<<<")
        #StrainStressColorPlot(strain[:,:,:,50], stress[:,:,:,50])
    mean_stiffness[iter_direction,:] = np.copy(np.mean(stress, axis=(1,2,3)))
    if (iter_direction > 2):
        mean_stiffness[iter_direction,:] = 0.5 * mean_stiffness[iter_direction,:]
mean_stiffness = np.transpose(mean_stiffness)

StrainStressColorPlot(strain[:,:,:,int(0.5 * size[2])], stress[:,:,:,int(0.5 * size[2])])
print("displacement")
print(np.min(displacement_field), np.max(displacement_field))
print("strain")
print(np.min(strain[0,:,:,:]), np.max(strain[0,:,:,:]))
print(np.min(strain[1,:,:,:]), np.max(strain[1,:,:,:]))
print(np.min(strain[2,:,:,:]), np.max(strain[2,:,:,:]))
print(np.min(strain[3,:,:,:]), np.max(strain[3,:,:,:]))
print(np.min(strain[4,:,:,:]), np.max(strain[4,:,:,:]))
print(np.min(strain[5,:,:,:]), np.max(strain[5,:,:,:]))
print("stress")
print(np.min(stress[0,:,:,:]), np.max(stress[0,:,:,:]))
print(np.min(stress[1,:,:,:]), np.max(stress[1,:,:,:]))
print(np.min(stress[2,:,:,:]), np.max(stress[2,:,:,:]))
print(np.min(stress[3,:,:,:]), np.max(stress[3,:,:,:]))
print(np.min(stress[4,:,:,:]), np.max(stress[4,:,:,:]))
print(np.min(stress[5,:,:,:]), np.max(stress[5,:,:,:]))
stress_divergence = Divergence(stress, step_x, step_y, step_z)
print("stress divergence")
print(np.min(stress_divergence[0,:,:,:]), np.max(stress_divergence[0,:,:,:]))
print(np.min(stress_divergence[1,:,:,:]), np.max(stress_divergence[1,:,:,:]))
print(np.min(stress_divergence[2,:,:,:]), np.max(stress_divergence[2,:,:,:]))
print(np.sqrt(np.mean(stress_divergence[0,:,:,:]**2.0 + stress_divergence[1,:,:,:]**2.0 + stress_divergence[2,:,:,:]**2.0)) / np.sqrt(np.mean(stress[0,:,:,:])**2.0 + np.mean(stress[1,:,:,:])**2.0 + np.mean(stress[2,:,:,:])**2.0))

step = np.array([step_x, step_y, step_z])
DeformationDataToVTK(Data.astype(np.int16), displacement_field, strain, stress, size, step, "name")

    
    
    