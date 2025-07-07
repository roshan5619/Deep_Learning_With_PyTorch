import torch
my_list=[[1,2,3],[4,5,6]]
tensor=torch.tensor(my_list)
print(tensor)

#TENSOR ATTRIBUTES
print(tensor.shape)#gives the output of list size

print(tensor.dtype)#gives the output of datatype of list

#To perform the tensor operations,the size of tensors should be same.

a=torch.tensor([[1,2],[2,3]])
b=torch.tensor([[2,3],[3,4]])

print(a+b)  #Addition
print(a-b)  #Substraction
print(a*b)  #Multiplication

print(a@b) #Matrix Multiplication i.e., (1st row of matrix a multiplied with 1st col of matrix b)
