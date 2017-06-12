import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 

    output_label=np.array([1,2,3,4,5])
    
    means=np.zeros((2,5))
    z_1=[]
    z_2=[]
    z_3=[]
    z_4=[]
    z_5=[]
    


    for i in range(0,y.shape[0]):
        #z=np.where(y==output_label[i])
        if(y[i]==output_label[0]):
            z_1.append(i)
            #means[0,0]=np.mean(X[i])
        if(y[i]==output_label[1]):
            z_2.append(i)
        if(y[i]==output_label[2]):
            z_3.append(i)
        if(y[i]==output_label[3]):
            z_4.append(i)
        if(y[i]==output_label[4]):
            z_5.append(i)
  
    
    X_1_1=[]
    X_1_2=[]
    X_2_1=[]
    X_2_2=[]
    X_3_1=[]
    X_3_2=[]
    X_4_1=[]
    X_4_2=[]
    X_5_1=[]
    X_5_2=[]
    
    for i in range(0,150):
        if (i in z_1):
            X_1_1.append(X[i][0])
            X_1_2.append(X[i][1])
            means[0][0]=np.mean(X_1_1)
            means[1][0]=np.mean(X_1_2)
        if (i in z_2):
            X_2_1.append(X[i][0])
            X_2_2.append(X[i][1])
            means[0][1]=np.mean(X_2_1)
            means[1][1]=np.mean(X_2_2)
        if (i in z_3):
            X_3_1.append(X[i][0])
            X_3_2.append(X[i][1])
            means[0][2]=np.mean(X_3_1)
            means[1][2]=np.mean(X_3_2)
        if (i in z_4):
            X_4_1.append(X[i][0])
            X_4_2.append(X[i][1])
            means[0][3]=np.mean(X_4_1)
            means[1][3]=np.mean(X_4_2)
        if (i in z_5):
            X_5_1.append(X[i][0])
            X_5_2.append(X[i][1])
            means[0][4]=np.mean(X_5_1)
            means[1][4]=np.mean(X_5_2)

   
    
    covmat=np.cov(X,rowvar=False)
    
    
    
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    
    output_label=np.array([1,2,3,4,5])
    means=np.zeros((2,5))
    z_1=[]
    z_2=[]
    z_3=[]
    z_4=[]
    z_5=[]
    for i in range(0,y.shape[0]):
        #z=np.where(y==output_label[i])
        if(y[i]==output_label[0]):
            z_1.append(i)
            
        if(y[i]==output_label[1]):
            z_2.append(i)
        if(y[i]==output_label[2]):
            z_3.append(i)
        if(y[i]==output_label[3]):
            z_4.append(i)
        if(y[i]==output_label[4]):
            z_5.append(i)
    X_1_1=[]
    X_1_2=[]
    X_2_1=[]
    X_2_2=[]
    X_3_1=[]
    X_3_2=[]
    X_4_1=[]
    X_4_2=[]
    X_5_1=[]
    X_5_2=[]
    
    for i in range(0,150):
        if (i in z_1):
            X_1_1.append(X[i][0])
            X_1_2.append(X[i][1])
            means[0][0]=np.mean(X_1_1)
            means[1][0]=np.mean(X_1_2)
        if (i in z_2):
            X_2_1.append(X[i][0])
            X_2_2.append(X[i][1])
            means[0][1]=np.mean(X_2_1)
            means[1][1]=np.mean(X_2_2)
        if (i in z_3):
            X_3_1.append(X[i][0])
            X_3_2.append(X[i][1])
            means[0][2]=np.mean(X_3_1)
            means[1][2]=np.mean(X_3_2)
        if (i in z_4):
            X_4_1.append(X[i][0])
            X_4_2.append(X[i][1])
            means[0][3]=np.mean(X_4_1)
            means[1][3]=np.mean(X_4_2)
        if (i in z_5):
            X_5_1.append(X[i][0])
            X_5_2.append(X[i][1])
            means[0][4]=np.mean(X_5_1)
            means[1][4]=np.mean(X_5_2)

    
   
    Labels=np.unique(y.reshape(y.size))
    
    covmats=[np.zeros((2,2))]*5
    for i in range(0,5): 
       covmats[i]=np.cov(X[y.reshape(y.size)==Labels[i]],rowvar=0)
    


    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    
    cov_inverse=inv(covmat)
    

    det_cov=det(covmat)
    
    step_0=((np.sqrt(2*np.pi))*(np.sqrt(det_cov)))
    
    pdf=np.zeros((Xtest.shape[0],5))
    for i in range(0,5):

       
        pdf[:,i] = np.exp(-0.5*np.sum((Xtest - means[:,i])*np.dot(cov_inverse, (Xtest - means[:,i]).T).T,axis=1))/(np.sqrt(np.pi*2)*(np.sqrt(det_cov)));
    
    acc = 100*np.mean((np.argmax(pdf,axis=1)+1) == ytest.reshape(ytest.size));
  
    

    ypred=np.argmax(pdf,axis=1)
    
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    
    pdf=np.zeros((Xtest.shape[0],5))
    for i in range(0,5):
        cov_inverse=inv(covmats[i])
        det_cov=det(covmats[i])
        pdf[:,i] = np.exp(-0.5*np.sum((Xtest - means[:,i])*np.dot(cov_inverse, (Xtest - means[:,i]).T).T,axis=1))/(np.sqrt(np.pi*2)*(np.sqrt(det_cov)));        


    acc = 100*np.mean((np.argmax(pdf,axis=1)+1) == ytest.reshape(ytest.size));
    ypred=np.argmax(pdf,axis=1)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
   
    X_transpose=np.transpose(X)
    X_w=np.dot(X_transpose,X)
    X_y=np.dot(X_transpose,y)
    X_w_inv=inv(X_w)
    w=np.dot(X_w_inv,X_y)
    
    
	                                                
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1    
    N= X.shape[1] 
    
    x_tran =np.transpose(X)
    x_mul= np.dot(x_tran,X)
    iden= np.identity(N)
    scal=iden*lambd

    term1= x_mul+scal
    x_inv= inv(term1)
    mul1=np.dot(x_inv,x_tran)

    w=np.dot(mul1,y)
                                                 
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    N=Xtest.shape[0]; # for the number of rows
    
    Xtest_transpose=np.transpose(Xtest); #converted from Nxd to dxN;

    w_X=np.dot(np.transpose(w),Xtest_transpose); # converted w from dx1 to 1xd finally getting 1xN
    w_X_transpose=np.transpose(w_X);
    loss=ytest-w_X_transpose; # for substraction the dimensions should be the same
    loss_sq=np.square(loss)
    loss_sum=np.sum(loss_sq);
    mse=loss_sum/N;
    

    
   
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda     
    w = w.reshape((w.shape[0],1))

    temp1=np.dot(X,w);
    
    temp2=y-temp1
    
    temp2_trans=np.transpose(temp2);
    temp3=np.dot(temp2_trans,temp2);
    w_trans=np.transpose(w);
    t4=np.dot(w_trans,w);
    t5=lambd*t4;
    error=temp3+t5;
    x_trans=np.transpose(X);
    t6=np.dot(x_trans,temp2);
    t7=-2*t6;
    t8=2*lambd*w; 
   
   
    error_grad=t7+t8;                                            
    return error.flatten(), error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    N=len(x)
    Xd=np.zeros(shape=(N,p+1))
    for i in range(0,N):
        for j in range(0,p+1):
            Xd[i][j]=np.power(x[i],j)
    # IMPLEMENT THIS METHOD
    return Xd

# Main script


# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA

means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)

print('LDA Accuracy = '+str(ldaacc))

# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))


# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()


# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_training=testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_training_i=testOLERegression(w_i,X_i,y)


print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))
print('MSE without intercept training '+str(mle_training))
print('MSE with intercept training'+str(mle_training_i))



# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1


fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()


# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)


fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

