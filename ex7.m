clear,clc;
fidin=fopen('twofeature.txt'); 
i=1;
apres=[];

while ~feof(fidin)
    tline = fgetl(fidin); 
    apres{i} = tline;
    i=i+1;
end
m0=length(apres);
for i=1:m0
    a=char(apres(i));
    if a(1)=='1'
        a=['+',a];
    end
    l=length(a);
    x0=sscanf(a(4:l),'%d:%f');
    lx=length(x0);
    for j=2:2:lx
        if(x0(j)<=0)
            break
        end
        x(i,x0(j-1))=x0(j);
    end
    y0=sscanf(a(1:2), '%d');
    y(i,1)=y0;
end

% plot
pos=(y==1);
neg=(y==-1);
figure;
plot(x(pos,1),x(pos,2),'ro');
hold on
plot(x(neg,1),x(neg,2),'b+');
m=length(y);
C=100;
f=-ones(1,m);
H=(y*y').*(x*x');
lb=zeros(1,m);
ub=C*ones(1,m);
Aeq=y';
beq=0;
alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub);
w=sum((alpha.*y).*x);

t=x*w';
max=-1000;
min=1000;
for i=1:m
    if(y(i)==-1&&t(i)>max)
        max=t(i);
    end
end
for i=1:m
    if(y(i)==1&&t(i)<min&&t(i)>max)
        min=t(i);
    end
end
b=-(max+min)/2;

% w1*x1+w2*x2+b=0
x1=[0:0.1:4.5];
x2=-((w(1)*x1)+b)./w(2);
hold on
plot(x1,x2,'-');
xlabel('feature1');
ylabel('feature2');
legend('data1','data2','boundary');

% solution
figure;
svm=svmtrain(x,y,'boxconstraint',1,'Showplot',true);


%%
clear,clc;
k=4000;
fidin=fopen('email_train-400.txt'); 
i=1;
apres=[];

while ~feof(fidin)
    tline = fgetl(fidin); 
    apres{i} = tline;
    i=i+1;
end
m0=length(apres);
x=zeros(m0,k);
y=zeros(m0,1);
for i=1:m0
    a=char(apres(i));
    if a(1)=='1'
        a=['+',a];
    end
    l=length(a);
    x0=sscanf(a(4:l),'%d:%f');
    lx=length(x0);
    for j=2:2:lx
        if(x0(j)<=0)
            break
        end
        x(i,x0(j-1))=x0(j);
    end
    y0=sscanf(a(1:2), '%d');
    y(i,1)=y0;
end
svm=svmtrain(x,y);

% plot
m=length(y);
C=1;
f=-ones(1,m);
H=(y*y').*(x*x');
lb=zeros(1,m);
ub=C*ones(1,m);
Aeq=y';
beq=0;

alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub);
w=sum((alpha.*y).*x);

max=-1000;
min=1000;
for i=1:m
    t=x(i,:)*w';
    if (y(i)==1&&t<min)
        min=t;
    elseif (y(i)==-1&&t>max)
        max=t;
    end
end
b=-(max+min)/2;

%
fidin=fopen('email_test.txt'); 
i=1;
apres=[];
while ~feof(fidin)
    tline = fgetl(fidin); 
    apres{i} = tline;
    i=i+1;
end
m0=length(apres);
tx=zeros(m0,k);
ty=zeros(m0,1);
y=zeros(m0,1);
for i=1:m0
    a=char(apres(i));
    if a(1)=='1'
        a=['+',a];
    end
    l=length(a);
    x0=sscanf(a(4:l),'%d:%f');
    lx=length(x0);
    for j=2:2:lx
        if(x0(j)<=0)
            break
        end
        tx(i,x0(j-1))=x0(j);
    end
    y0=sscanf(a(1:2), '%d');
    y(i,1)=y0;
end
ty=tx*w'+b;
accuracy=sum(((ty>=0)*2-1)==y)/m0

ylabel=svmclassify(svm,tx,'Showplot',true);
accuracy0=sum(((ylabel>=0)*2-1)==y)/m0
