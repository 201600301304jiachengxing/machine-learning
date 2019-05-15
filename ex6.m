clear,clc;
L={'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality'};
M=csvread('ex6Data.csv',1);
m=length(M);
k=10;
kf=k;
N=length(L);
data=crossvalind('Kfold',m,k);
p=0.0001;
number=100;
 
for i=1:k
    itest=(data==i);
    itrain=(data~=i);
    dtest=M(itest,:);
    dtrain=M(itrain,:);
    D0=[1:N];
    M0=dtrain;
    j=0;
    clear t;
    t(1)=buildtree(D0,M0);
    while(j~=length(t))
        j=j+1;
        if length(t(j).num)==0
            t(j).num=0;
            continue; 
        elseif(t(j).num~=0&&t(j).rate>=p)
            Mt=t(j).M;
            Mi=t(j).index;
            Md=t(j).D;
            if(Mi>number&&Mi<length(Mt)-number)
                Ml=Mt(1:Mi,:);
                t(j*2)=buildtree(Md,Ml);
                Mr=Mt(Mi+1:length(Mt),:);
                t(j*2+1)=buildtree(Md,Mr);
            else
                t(j*2).num=0;
                t(j*2+1).num=0;
            end
        elseif(t(j).num~=0&&t(j).rate<p)
            t(j*2).num=0;
            t(j*2+1).num=0;
        end
    end
    for k=1:j
        if(t(k).num~=0&&k*2<=j&&t(k*2).num==0)
            Ml=t(k).M;
            pos=length(Ml(Ml(:,N)==1));
            neg=length(Ml(Ml(:,N)==0));
            if(pos>neg)
                t(k).label=1;
            else
                t(k).label=0;
            end
        end
    end
    a=length(dtest);
    sum1=0;
    for k=1:a
        g=1;
        while(t(g*2).num~=0)
            num=t(g).num;
            thres=t(g).threshold;
            if dtest(k,num)<=thres
                g=g*2;
            else
                g=g*2+1;
            end
        end
        if t(g).label==dtest(k,N)
            sum1=sum1+1;
        end
    end
    ac(i)=sum1/length(dtest)
end
rac=sum(ac)/kf;
plotree(t,L);
 
%% plot tree
function []=plotree(t,L)
l=length(t);
px(1)=0;
py(1)=0;
plot(px(1),py(1),'bo');
text(px(1)+5,py(1),L(t(1).num));
text(px(1)+5,py(1)-2,"<"+num2str(t(1).threshold));
for k=2:l
    if t(k).num~=0
        fm=floor(k/2);
        if(mod(k,2)==0)
            px(k)=px(fm)-2^(10-log2(fm));
            py(k)=py(fm)-1.4^(10-log2(fm));
            text(px(k)-150,py(k),L(t(k).num));
            text(px(k)-150,py(k)-2,"<"+num2str(t(k).threshold));
            if length(t(k).label)~=0
                text(px(k)-150,py(k)-4,"CLASS "+num2str(t(k).label));
            end
        else
            px(k)=px(fm)+2^(10-log2(fm));
            py(k)=py(fm)-1.4^(10-log2(fm));
            text(px(k)+50,py(k),L(t(k).num));
            text(px(k)+50,py(k)-2,"<"+num2str(t(k).threshold));
            if length(t(k).label)~=0
                text(px(k)+50,py(k)-4,"CLASS "+num2str(t(k).label));
            end
        end
        hold on
        plot(px(k),py(k),'bo');
        hold on;
        plot([px(fm):(px(k)-px(fm))/10000:px(k)],[py(fm):(py(k)-py(fm))/10000:py(k)],'b-');
    end
end
end
 
%% build tree
function [tree]=buildtree(D,M)
[g,class,th,index,rate]=gain(D,M);
M0=sortrows(M,class);
D0=setdiff(D,[class]);
 
tree.num=class;
tree.threshold=th;
tree.rate=rate;
tree.gain=g;
tree.D=D0;
tree.M=M0;
tree.index=index;
end
 
%% 信息增益
function [g,class,th,index,rate]=gain(D,M)
m=length(M);
n=length(D);
pos=length(M(M(:,D(n))==1))/m;
neg=length(M(M(:,D(n))==0))/m;
Ent=-(pos*log(pos)+neg*log(neg));
%
for i=1:n-1
    pos=D(i);
    Ms=sortrows(M,pos);
    Enti(pos)=1;
    for j=1:m-1
        if(Ms(j,n)~=Ms(j+1,n))
            Ms1=[];
            Ms1=Ms(1:j,:);
            pos1=length(Ms1(Ms1(:,D(n))==1))/(j);
            neg1=length(Ms1(Ms1(:,D(n))==0))/(j);
            Ent1=-(pos1*log(pos1)+neg1*log(neg1));
            Ms2=[];
            Ms2=Ms(j+1:m,:);
            pos2=length(Ms2(Ms2(:,D(n))==1))/(m-j);
            neg2=length(Ms2(Ms2(:,D(n))==0))/(m-j);
            Ent2=-(pos2*log(pos2)+neg2*log(neg2));
            e=(j/m)*Ent1+(1-j/m)*Ent2;
            if e<Enti(pos)
                Enti(pos)=e;
                t(pos)=(Ms(j,pos)+Ms(j+1,pos))/2;
                y(pos)=j;
            end
        end
    end
end
gs=1;
class=1;
index=1;
th=0;
for i=1:n-1
    pos=D(i);
    if Enti(pos)<gs
        gs=Enti(pos);
        class=pos;
        index=y(pos);
        th=t(pos);
    end
end
g=Ent-gs;
rate=g/Ent;
end







