clear,clc;
%% show the data
x=load('ex1x.dat');
y=load('ex1y.dat');
figure
plot(x,y,'o');
xlabel('age in years');
ylabel('height in meters');

m=length(y);
X=[ones(m,1),x];

%% show the train(y=Xw,w=inv(X'X)*X')
theta=pinv(X)*y
(X*theta-y)'*(X*theta-y)/(2*m)
hold on;
plot(X(:,2),X*theta,'-x');

%% gradient descent
theta=zeros(2,1);
alpha=0.07;
for k=1:1500
    theta=theta-alpha*sum((X*theta-y).*X)'/(m);
    if k==1
        theta
    end
end
hold on;
plot(X(:,2),X*theta,'-o');
legend('training data','linear regression','GD')

%% loss=1/2*sum(x*theta-y)^2
Jtheta=zeros(100,100);
theta0=linspace(-3,3,100);
theta1=linspace(-1,1,100);

for p=1:length(theta0)
    for q=1:length(theta1)
        t=[theta0(p);theta1(q)];
        Jtheta(p,q)=(X*t-y)'*(X*t-y)/(2*m);
    end
end
figure
surf(theta0,theta1,Jtheta')
xlabel('\theta_0');
ylabel('\theta_1');

figure
contour(theta0,theta1,Jtheta',1000)
xlabel('\theta_0');
ylabel('\theta_1');

%% pre
y1=[1,3.5]*theta
y2=[1,7.0]*theta





