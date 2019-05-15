clear,clc;
%% load data
xb=load('ex3blue.dat');
xg=load('ex3green.dat');
xr=load('ex3red.dat');

%% bi-classification
figure
plot(xb(:,1),xb(:,2),'bx');
hold on 
plot(xr(:,1),xr(:,2),'rx');

mb=mean(xb);
mr=mean(xr);
Sw=(xb-mb)'*(xb-mb)+(xr-mr)'*(xr-mr);
w=inv(Sw)*(mb-mr)';
w=abs(w)/sqrt(w'*w);

x=[0:0.01,10];
y=(w(2)/w(1))*x;
hold on
plot(x,y,'-d');

zb=xb*w*w';
zr=xr*w*w';
plot(zb(:,1),zb(:,2),'bx');
hold on 
plot(zr(:,1),zr(:,2),'rx');
xlabel('feature 1');
ylabel('feature 2');
legend('data blue','data red','LDA','line blue','line red');
title('bi-classification');

%% distribution
yb=xb*w;
yr=xr*w;
sb=std(yb);
sr=std(yr);
ub=mean(yb);
ur=mean(yr);
tb=[ub-4*sb:0.01:ub+4*sb];
tr=[ur-4*sr:0.01:ur+4*sr];
figure
plot(tb,exp(-(tb-ub).*(tb-ub)/(2*sb^2))/(sb*sqrt(2*pi)),'b-');
hold on
plot(tr,exp(-(tr-ur).*(tr-ur)/(2*sr^2))/(sr*sqrt(2*pi)),'r-');
title('distribution of data');
legend('data blue','data red');

%% multi-classification
figure
plot(xb(:,1),xb(:,2),'bx');
hold on 
plot(xr(:,1),xr(:,2),'rx');
hold on
plot(xg(:,1),xg(:,2),'gx');

mb=mean(xb);
mr=mean(xr);
mg=mean(xg);
nb=length(xb);
nr=length(xr);
ng=length(xg);
na=nb+nr+ng;
ma=(nb*mb+nr*mr+ng*mg)/na;

St=(xb-ma)'*(xb-ma)+(xr-ma)'*(xr-ma)+(xg-ma)'*(xg-ma);
Sw=(xb-mb)'*(xb-mb)+(xr-mr)'*(xr-mr)+(xg-mg)'*(xg-mg);
Sb=St-Sw;

[vec,val]=eig(inv(Sw)*Sb);
lamda=max(diag(val));
W=vec(:,find(diag(val==lamda)));

x=[0:0.01,10];
y=(W(2)/W(1))*x;
hold on
plot(x,y,'-d');

zb=xb*W*W';
zr=xr*W*W';
zg=xg*W*W';
plot(zb(:,1),zb(:,2),'bx');
hold on 
plot(zr(:,1),zr(:,2),'rx');
hold on
plot(zg(:,1),zg(:,2),'gx');
xlabel('feature 1');
ylabel('feature 2');
legend('data blue','data red','data green','LDA','line blue','line red','line green');
title('multi-classification');

%% distribution
yb=xb*W;
yr=xr*W;
yg=xg*W;
sb=std(yb);
sr=std(yr);
sg=std(yg);
ub=mean(yb);
ur=mean(yr);
ug=mean(yg);
tb=[ub-4*sb:0.01:ub+4*sb];
tr=[ur-4*sr:0.01:ur+4*sr];
tg=[ug-4*sg:0.01:ug+4*sg];
figure
plot(tb,exp(-(tb-ub).*(tb-ub)/(2*sb^2))/(sb*sqrt(2*pi)),'b-');
hold on
plot(tr,exp(-(tr-ur).*(tr-ur)/(2*sr^2))/(sr*sqrt(2*pi)),'r-');
hold on
plot(tg,exp(-(tg-ug).*(tg-ug)/(2*sg^2))/(sg*sqrt(2*pi)),'g-');
title('distribution of data');
legend('data blue','data red','data green');



