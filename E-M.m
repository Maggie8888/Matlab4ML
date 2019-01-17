function [h,M,Q] = EMG1(imagefile,k,lambda)
%import imagefire and convert to 2D matrix with 3 colomns
[img,cmap]= imread(imagefile);
img_rgb =ind2rgb(img,cmap);
img_double=im2double(img_rgb);
X=reshape(img_double, [],3);
N=size(X,1);
D=size(X,2);

% Set the number of class as k={4,8,12}
% M is the centroid matirx of k class by d dimension after 2 iteration of
% kMeans clustering
[idx, M]=kmeans(X,k,'MaxIter',2,'EmptyAction','singleton');
% let i denote the index of class, so we calculate the covariance of each
% class and store it in 3d by 3d by k matrix S(:,:,i)
%Calculate prior probability of class i as pi(i)

for i= 1:k
    S(:,:,i)=cov(X(idx(:)==i,:));
    S_inv=inv(S(:,:,i));
    pi(i)= sum(idx(:)==i)/N;
        for j=1:N
           p(:,i)=(det(S(:,:,i)))^(-0.5)*exp((-0.5)*(X(j,:)-M(i,:))*S_inv*transpose(X(j,:)-M(i,:)));
        end
end

for iter=1:100
    %h is N by k matrix
    for i=1:k
        S(:,:,i)=cov(X(idx(:)==i,:));
        S_inv=inv(S(:,:,i));
        for j=1:N
        % First we calculate the numerator of h
            h(j,i)=pi(i).*(det(S(:,:,i)))^(-0.5)*exp((-0.5)*(X(j,:)-M(i,:))*S_inv*transpose(X(j,:)-M(i,:)));

        end
    end
    %Note we need to devide each row of h by sum of that row. So we use function bsxfun
    % to right divide the sum of each row.)
    h=bsxfun(@rdivide, h, sum(h,2));
    %In order to simplify, introduce Ni, 1 by k matrix
    Ni=sum(h);
   
    for i= 1:k
        for j=1:N
        % First we calculate the numerator of h
           p(:,i)=(det(S(:,:,i)))^(-0.5)*exp((-0.5)*(X(j,:)-M(i,:))*S_inv*transpose(X(j,:)-M(i,:)));
        end
    end

    p(p==0)=0.0000001;
    pi(pi==0)=0.0000001;
    Q(iter,1)=sum(h)*transpose(log(pi))+sum(sum(h.*log(p)))-(lambda/2)*(sum(sum(diag(S_inv))));
    %
    M=bsxfun(@rdivide,transpose(h)*X, transpose(Ni));
    % S(:,:,i) is a d by d matrix
    for i=1:k
        % first calculate the numerator of S
        S(:,:,i)=transpose(X-repmat(M(i,:),N,1))*((X-repmat(M(i,:),N,1)).*repmat(h(:,i),1,3));
        % Then divided by denominator
        S(:,:,i)=S(:,:,i)./Ni(i);
    end
    


     %Calculate the expectation after M step for each iteration  
    p(p==0)=0.0000001;
    pi(pi==0)=0.0000001; 
    Q(iter,2)=sum(h)*transpose(log(pi))+sum(sum(h.*log(p)))-(lambda/2)*(sum(sum(diag(S_inv))));
end
[ ~, row_argmax ] = max(h,[],2);

Compress=zeros(N,3);
for j=1:N
    Compress(j,:)=M(row_argmax(j),:);
end
Compress=reshape(Compress, size(img_rgb,1),size(img_rgb,2),3);        
figure
imagesc(Compress);
ti = sprintf('lambda = %g',lambda);
title(ti);

figure
x=0.5:0.5:100;
x1=0.5:1:99.5;
x2=1:1:100;
Q_line=transpose(Q);
plot(x,Q_line(:));
hold all;
scatter(x1,Q(:,1),'.','r');
scatter(x2,Q(:,2),'.','k');
ti = sprintf('lambda = %g',lambda);
title(ti);
hold off;
end

