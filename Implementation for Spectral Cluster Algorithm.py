import numpy as np 
import matplotlib.pyplot as plt


#load data
data_file = "SpectData.txt"
data = np.loadtxt(data_file, dtype= float)
print(data.shape)

#Calculate Weight matrix 
def Weight_Matrix(std_dev):
    weight_list=[]
    for i in range(len(data)):
        for n in range (len(data)):
            dis_square= np.dot(data[i]-data[n], data[i]-data[n]) 
            weight=np.exp(-(dis_square/(2*(std_dev**2))))
            weight_list.append(weight)
    weight_array=np.array(weight_list) 
    W_matrix=weight_array.reshape((len(data),len(data)))  
    np.fill_diagonal(W_matrix,0)
    return  W_matrix



#Calculate Diagonal Matrix
def Diagonal_Matrix_Fn(W_matrix):
    sum_weight=W_matrix.sum(axis=1)  
    Diagonal_Matrix= np.zeros((len(data),len(data))) 
    np.fill_diagonal(Diagonal_Matrix,sum_weight) 
    return Diagonal_Matrix

# Calculate Laplacian Matrix
def Laplacian_Matrix_Fn(W_matrix,Diagonal_Matrix):
    Laplacian_Matrix=np.subtract(Diagonal_Matrix,W_matrix)
    return  Laplacian_Matrix

#Calculate Eigen Vectors and Eigen Values
def Eigen_Values_Eigen_Vectors_Fn(Laplacian_Matrix):
    eig_value ,eig_vector=np.linalg.eig(Laplacian_Matrix)
    eig_value_sort =sorted(eig_value)
    index=(list(eig_value).index(eig_value_sort[1])) 
    eig_vectors= eig_vector.T 
    return eig_vectors ,index


def clusters_plot(eig_vectors,index,std_dev): 
    color=[]
    for i in range (len(data)):
        if  eig_vectors[index][i]>0:
            color.append('r')
        else:
            color.append('g')
    plt.scatter(data[:,0],data[:,1],  color= color)
    plt.title('Spect_'+str(std_dev))
    plt.savefig('Spect_'+str(std_dev)+'.jpg')
    plt.show() 


#Plot Spectral Clustering with any given Aplpha
def plot_spectral_cluster_give_alpha(std_dev):
    W_matrix=Weight_Matrix(std_dev)
    Diagonal_Matrix=Diagonal_Matrix_Fn(W_matrix)   
    Laplacian_Matrix=Laplacian_Matrix_Fn(W_matrix,Diagonal_Matrix)
    eig_vectors,index=Eigen_Values_Eigen_Vectors_Fn(Laplacian_Matrix)
    clusters_plot(eig_vectors,index,std_dev)
     
## Spectral clustering plot for Sigma 0.01
std_dev = 0.01
plot_spectral_cluster_give_alpha(std_dev)
## Spectral clustering plot for Sigma 0.05
std_dev = 0.05
plot_spectral_cluster_give_alpha(std_dev)
## Spectral clustering plot for Sigma 0.1
std_dev = 0.1
plot_spectral_cluster_give_alpha(std_dev)
