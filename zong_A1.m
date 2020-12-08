tic
clear;
load('abu-airport-3.mat');
[a1,a2,a3]=size(data);
load A1Xzresult;
load A1XSresult
load A1LH 
xx=0.01*xxzr+0.0005*xxsr+1000000000*kb1;
xxs=reshape(xx,a1*a2,[]);

for i=1:a1*a2
    xx0(i)=xxs(i,:);
end
xxz=reshape(xx0,a1,a2);
[a1,a2]=size(map);
kb0=reshape(xx0,1,a1*a2);
pp=reshape(kb0,a1,a2);

output=reshape(pp,a1*a2,1);
YT=map;
test_targets=reshape(YT,a1*a2,1);
[A,I]=sort(output);
M=0;N=0;
for i=1:length(output)
    if(test_targets(i)==1)
        M=M+1;
    else
        N=N+1;
    end
end
sigma=0;
for i=M+N:-1:1
    if(test_targets(I(i))==1)
        sigma=sigma+i;
    end
end
result=(sigma-(M+1)*M/2)/(M*N)


