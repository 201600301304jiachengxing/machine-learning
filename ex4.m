clear,clc;
%% data load
x=load('ex4x.dat');
y=load('ex4y.dat');
m=length(y);
X=[ones(m,1),x];

%% show the data
pos=find(y==1);
neg=find(y==0);
figure;
plot(X(pos,2),X(pos,3),'+');
hold on
plot(X(neg,2),X(neg,3),'o');
xlabel('exam 1 score');
ylabel('exam 2 scroe');

%% train(newton's method)
e=1e-9;
theta=zeros(3,1);
loss=zeros(1,5);
num=1;
loss(1,num)=inf;
% repeat
num=num+1;
h=1./(1+exp(-X*theta));
H=(1/m)*X'*(h.*(1-h).*X);
hinv=inv(H);
loss(1,num)=-(1/m)*sum((1-y).*log(1-h)+y.*log(h));
theta=theta-hinv*(1/m)*(X'*(h-y));
while abs(loss(1,num)-loss(1,num-1))>e
    num=num+1;
    h=1./(1+exp(-X*theta));
    H=(1/m)*X'*(h.*(1-h).*X);
    hinv=inv(H);
    loss(1,num)=-(1/m)*sum((1-y).*log(1-h)+y.*log(h));
    theta=theta-hinv*(1/m)*(X'*(h-y));
end
% theta*X=0,show the decision boundary(newton's method)
x1=[15:0.1:65];
x2=(theta(1)+theta(2)*x1)/(-theta(3));
hold on
plot(x1,x2,'-');
legend('admitted','not admitted','boundary');

%% plot
figure
plot([1:num],loss,'-');
xlabel('iteration');
ylabel('loss');
legend('newton');

%% pre
x0=[1,20,80];
y0=1/(1+exp(-x0*theta));

%% show the result
theta
1-y0
num









