tic
clear;
load A1LR;
XXLR=mat2gray(A1LR);
[a1,a2,a3]=size(XXLR);
Y=XXLR;
inner_w=1; 
d3=a3*inner_w*inner_w;
half_in=(inner_w-1)/2;
center_in=(inner_w+1)/2; 
A1LRR=zeros(a1,a2,d3);
Y_ex=zeros(a1+inner_w-1,a2+inner_w-1,a3);
for k=1:a3
    Y_ex(:,:,k)=padarray(Y(:,:,k),[half_in,half_in],'replicate');
end
for i=center_in:a1+half_in
    for j=center_in:a2+half_in
        current_w=Y_ex(i-half_in:i+half_in,j-half_in:j+half_in,:);
        detect=current_w(center_in-half_in:center_in+half_in,center_in-half_in:center_in+half_in,:);
        A1LRR(i-half_in,j-half_in,:)=reshape(detect,1,[]);
    end
end
A1LRRR=reshape(A1LRR,a1*a2,[]);



%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
%train dbn
dbn.sizes = [50 30 50 d3];
opts.numepochs = 1;
opts.batchsize = 100;
opts.momentum  = 0;
opts.alpha     = 1;
dbn = dbnsetup(dbn, A1LRRR, opts);
dbn = dbntrain(dbn, A1LRRR, opts);
%unfold dbn to nn
nn = dbnunfoldtonn(dbn, d3);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 100;
%nn = nntrain(nn, Z5SRRR, Z5SRRR, opts);
[er, bad] = nntest(nn, A1LRRR, A1LRRR);
%assert(er < 0.10, 'Too big error');
x1=dbn.rbm{1,1}.W';
x1b=dbn.rbm{1,1}.b';
x2=dbn.rbm{1,2}.W';
x2b=dbn.rbm{1,2}.b';
x3=dbn.rbm{1,3}.W';
x3b=dbn.rbm{1,3}.b';
x4=dbn.rbm{1,4}.W';
x4b=dbn.rbm{1,4}.b';
%x=P2SRRR*x1+dbn.rbm{1,1}.b';
%x=(((P2SRRR*x1+x2b)*x2+x3b)*x3+x4b)*x4+x1b;
XL=A1LRRR*x1*x2;
save A1XL XL
x=A1LRRR*x1*x2*x3*x4;
xx=reshape(x,a1,a2,[]);

xxs=reshape(A1LRRR,a1*a2,[]);
for i=1:a1*a2
    xx0(i)=norm(xxs(i,:)-x(i,:))^2;
end

xxlr=reshape(xx0,a1,a2);



save A1XLresult xxlr;
kbs0=xxlr;
figure,imshow(xxlr,[])   %  Visualize the RBM weights

load('abu-airport-3.mat');
[a1,a2]=size(map);
kb0=reshape(kbs0,1,a1*a2);
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


