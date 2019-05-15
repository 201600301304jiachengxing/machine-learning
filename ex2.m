clear,clc;
%% load data
x=load('ex2x.dat');
y=load('ex2y.dat');

m=length(y);
X=[ones(m,1),x];

%% process
sig=std(X);
mu=mean(X);
X(:,2)=(X(:,2)-mu(2))/sig(2);
X(:,3)=(X(:,3)-mu(3))/sig(3);

%% pre-data
x_pre=[1 1650 3];
x_pre(2)=x_pre(2)-mu(2)/sig(2);
x_pre(3)=x_pre(3)-mu(3)/sig(3);
y_pre=[0 0 0 0];

%% learning rate
theta0=rands(3,1);
num=50;

% group 1
alpha=1.0;
Jtheta=zeros(num,1);
theta=theta0;
for k=1:num
    Jtheta(k)=(X*theta-y)'*(X*theta-y)/(2*m);
    theta=theta-alpha*sum((X*theta-y).*(X))'/m;
end
figure
plot(0:num-1,Jtheta,'b-','LineWidth',2);
y_pre(1)=x_pre*theta;
Jtheta(num)
theta

% group 2
alpha=0.5;
Jtheta=zeros(num,1);
theta=theta0;
for k=1:num
    Jtheta(k)=(X*theta-y)'*(X*theta-y)/(2*m);
    theta=theta-alpha*sum((X*theta-y).*(X))'/m;
end
hold on
plot(0:num-1,Jtheta,'r-','LineWidth',2);
y_pre(2)=x_pre*theta;
Jtheta(num)

% group 3
alpha=0.2;
Jtheta=zeros(num,1);
theta=theta0;
for k=1:num
    Jtheta(k)=(X*theta-y)'*(X*theta-y)/(2*m);
    theta=theta-alpha*sum((X*theta-y).*(X))'/m;
end
hold on
plot(0:num-1,Jtheta,'k-','LineWidth',2);
y_pre(3)=x_pre*theta;
Jtheta(num)

% group 4
alpha=0.1;
Jtheta=zeros(num,1);
theta=theta0;
for k=1:num
    Jtheta(k)=(X*theta-y)'*(X*theta-y)/(2*m);
    theta=theta-alpha*sum((X*theta-y).*(X))'/m;
end
hold on
plot(0:num-1,Jtheta,'g-','LineWidth',2);
y_pre(4)=x_pre*theta;
Jtheta(num)

% plot
xlabel('number of iterations');
ylabel('cost J');
legend('lr=1.0','lr=0.5','lr=0.2','lr=0.1')
title('loss of different learning-rate')

%% train(y=Xw,w=inv(X'X)*X')
theta0=pinv(X)*y
real_y_pre=x_pre*theta0;

%% show the result
y_pre
real_y_pre