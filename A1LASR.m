clear;tic;
load('abu-airport-1.mat')
X0=mat2gray(data);
[a,b,c]=size(X0);
X=reshape(X0,a*b,c);
rank=5;
card=1e+1;
power=0;
isize=[a,b];
[L,LR,map,S,RMSE,error]=GoDec(X,rank,card,power);
G=X-L-S;
figure;
i=10;
    subplot(1,4,1);imagesc(reshape(X(:,i),isize));colormap(gray);axis image;axis off;title('X(Sample)');
    subplot(1,4,2);imagesc(reshape(L(:,i),isize));colormap(gray);axis image;axis off;title('L(Low-rank)');
    subplot(1,4,3);imagesc(reshape(S(:,i),isize));colormap(gray);axis image;axis off;title('S(Sparse)');
    subplot(1,4,4);imagesc(reshape(G(:,i),isize));colormap(gray);axis image;axis off;title('G(Noise)');
 XXLR=reshape(L,a,b,[]);
 XXSR=reshape(S,a,b,[]);
 A1SR=XXSR;
 A1LR=XXLR;
 save A1LR A1LR
 save A1SR A1SR
 toc;
    