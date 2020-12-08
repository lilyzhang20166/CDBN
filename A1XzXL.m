clear;
load('abu-airport-3.mat');
load A1Xz;
load A1XL
[a1,a2,b3]=size(data);
Xz=reshape(Xz,a1,a2,[]);
Y=mat2gray(Xz);
XL=reshape(XL,a1,a2,[]);
YB=mat2gray(XL);
[a1,a2,a3]=size(Xz);
%processing window
outer_w=7;
inner_w=3;
center=(outer_w+1)/2; 
half_w=center-1;
window=true(7);
background=window;
background(center-1:center+1,center-1:center+1)=0;
detectionarea=~background;
background=reshape(background,[],1);
background_N=outer_w^2-inner_w^2;

% original data boundary extension
Y_ex=zeros(a1+outer_w-1,a2+outer_w-1,a3);
YB_ex=zeros(a1+outer_w-1,a2+outer_w-1,a3);
for k=1:a3
    Y_ex(:,:,k)=padarray(Y(:,:,k),[half_w,half_w],'replicate');
    YB_ex(:,:,k)=padarray(YB(:,:,k),[half_w,half_w],'replicate');
end

%%%%%%%%%produce dictionery%%%%%%%%
C=zeros(background_N,a1,a2);
Bq=zeros(a3,a1,a2);
for i=center:a1+half_w
    for j=center:a2+half_w
        current_w=Y_ex(i-half_w:i+half_w,j-half_w:j+half_w,:);
        currentB_w=YB_ex(i-half_w:i+half_w,j-half_w:j+half_w,:);
        currentB_wd=reshape(currentB_w,[],a3);
        detect=reshape(current_w(center,center,:),1,[]);
        n=1;
        background_data=zeros(background_N,a3);
        for m=1:outer_w^2
            if background(m)
                background_data(n,:)=currentB_wd(m,:);
                n=n+1;
            end
        end
        
        background_data1=reshape(background_data,a3,[]);
        detect1=reshape(detect,a3,1);
   
 %%%%%%%sparse detect%%%%%%%%%%%%%%%%
              lm=1e-2;
        A=pinv(background_data1'*background_data1+lm*eye(background_N))*background_data1'*detect1;
        C(:,i-half_w,j-half_w)=A;
    end
end
CC=reshape(C,background_N,a1*a2);
kb0=zeros(1,a1*a2);
for i=1:a1*a2
kb0(i)=sum((norm(CC(:,i)-sum(CC(:,i))/(background_N)))^2)/background_N;
end
U5resultCRD=kb0;
%save P2CRD35 P2resultCRD;
kb1=reshape(kb0,a1,a2);
save A1LH kb1
kb2=sort(kb0,'descend');
kb3=kb2(48);
ppb=zeros(a1,a2);
for i=1:a1
    for j=1:a2
        if(kb1(i,j)>=kb3)
            ppb(i,j)=255;
        else
            ppb(i,j)=0;
        end
    end
end
figure,imshow(ppb,[]); 


[a1,a2]=size(map);

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
toc;