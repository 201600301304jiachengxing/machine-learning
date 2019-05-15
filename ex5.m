clear,clc;
%% liner data load and preprocess
x=load('ex5Linx.dat');
y=load('ex5Liny.dat');
figure
plot(x,y,'o','MarkerFaceColor','r');

m=length(y);
X=[ones(m,1),x,x.^2,x.^3,x.^4,x.^5];
I0=eye(6);
I0(1,1)=0;

%% loss=(sum((X*theta-y).^2)+lamda*theta'*I0*theta)/2m
% lamda=0
lamda=0;
theta=inv(X'*X+lamda*I0)*X'*y;
hold on
x0=[-1:0.01:1]';
m0=length(x0);
x0=[ones(m0,1),x0,x0.^2,x0.^3,x0.^4,x0.^5];
plot(x0(:,2),x0*theta,'-');

% lamda=1
lamda=1;
theta=inv(X'*X+lamda*I0)*X'*y;
hold on
x0=[-1:0.01:1]';
m0=length(x0);
x0=[ones(m0,1),x0,x0.^2,x0.^3,x0.^4,x0.^5];
plot(x0(:,2),x0*theta,'-');

% lamda=10
lamda=10;
theta=inv(X'*X+lamda*I0)*X'*y;
hold on
x0=[-1:0.01:1]';
m0=length(x0);
x0=[ones(m0,1),x0,x0.^2,x0.^3,x0.^4,x0.^5];
plot(x0(:,2),x0*theta,'-');

legend('training data','lamda=0','lamda=1','lamda=10');

%% log data load and preprocess
clear,clc;
x=load('ex5Logx.dat');
y=load('ex5Logy.dat');
m=length(y);
pos=find(y==1);
neg=find(y==0);
figure
plot(x(pos,1),x(pos,2),'+');
hold on
plot(x(neg,1),x(neg,2),'o');
xlabel('data1');
ylabel('data2');

%% loss=-(1/m)*(ylog(h)+(1-y)log(1-h))+(2/m)*lamda*theta(2:n)^2
I=eye(28);
I(1,1)=0;
X=map_feature(x(:,1),x(:,2));

%% regulization
num=500;
%% lamda=0
lamda=0;
theta=zeros(28,1);
for j=1:num
    h=1./(1+exp(-X*theta));
    H=(1/m)*(X'*(h.*(1-h).*X)+lamda*I);
    invH=inv(H);
    theta=theta-invH*(1/m)*(X'*(h-y)+lamda*I*theta);
end
% plot
u=linspace(-1,1.5,200);
v=linspace(-1,1.5,200);
z=zeros(length(u),length(v));
for j=1:length(u)
    for k=1:length(v)
        z(j,k)=map_feature(u(j),v(k))*theta;
    end
end
hold on
contour(u,v,z',[0,0],'linewidth',1);

%% lamda=1
lamda=1;
theta=zeros(28,1);
for j=1:num
    h=1./(1+exp(-X*theta));
    H=(1/m)*(X'*(h.*(1-h).*X)+lamda*I);
    invH=inv(H);
    theta=theta-invH*(1/m)*(X'*(h-y)+lamda*I*theta);
end
% plot
u=linspace(-1,1.5,200);
v=linspace(-1,1.5,200);
z=zeros(length(u),length(v));
for j=1:length(u)
    for k=1:length(v)
        z(j,k)=map_feature(u(j),v(k))*theta;
    end
end
hold on
contour(u,v,z',[0,0],'linewidth',2);

%% lamda=10
lamda=10;
theta=zeros(28,1);
for j=1:num
    h=1./(1+exp(-X*theta));
    H=(1/m)*(X'*(h.*(1-h).*X)+lamda*I);
    invH=inv(H);
    theta=theta-invH*(1/m)*(X'*(h-y)+lamda*I*theta);
end
% plot
u=linspace(-1,1.5,200);
v=linspace(-1,1.5,200);
z=zeros(length(u),length(v));
for j=1:length(u)
    for k=1:length(v)
        z(j,k)=map_feature(u(j),v(k))*theta;
    end
end
hold on
contour(u,v,z',[0,0],'linewidth',4);

legend('positive','negative','lamda=0','lamda=1','lamda=10');









