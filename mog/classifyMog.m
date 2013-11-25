% Computes probability P(d = 1|x) and P(d = 2|x), where d=1 means 
% the given image is a 2, and d=2 means the given image is a 3
%
% Input:
%
%   Model parameters:
%
%    pD1(k) = prior probability of the kth cluster for d=1 model
%    muD1(n,k) = the nth component of the mean for the kth cluster for d=1 model
%    varyD1(j,k) = variance of the jth dimension in the kth cluster for d=1 model
%
%    pD2(k) = prior probability of the kth cluster for d=2 model
%    muD2(n,k) = the nth component of the mean for the kth cluster for d=2 model
%    varyD2(j,k) = variance of the jth dimension in the kth cluster for d=2 model
%
%    nD1 = number of cases where d = 1
%    nD2 = number of cases where d = 2
%
%   Data:
%
%    x(n,t) = the nth input for the tth training case
%  
% Output:
%
%    pD1givenX(t) = likelihood that d = 1 given x for the tth training case
%    pD2givenX(t) = likelihood that d = 2 given x for the tth training case

function [pi, class] = classifyMog(pD,muD,varyD,n,x) % n(i) is # of class i

    probDi = zeros(1, 7);
    pXgivenDi = zeros(size(x, 2), 7);
    pDigivenX = zeros(size(x, 2), 7);
    
    for i = 1:7
        
        % find P(d = i)
        probDi(i) = n(i) / sum(n);
        
        % find p(x | d = 1)
        pXgivenDi(:, i) = mogLogProb( ...
                        pD(:, i), ...
                        muD(:, :, i), ...
                        varyD(:, :, i), ...
                        x);

        % find p(d = 1|x)
        pDigivenX(:, i) = (1 / probDi(i)) * pXgivenDi(:, i);
    end
    
    pi = pDigivenX;
    
    [~, I] = max(pDigivenX, [], 2);
    theClass = I;
    
    class = theClass;
    %theClass = zeros(size(x, 2));
    
    %for i = 1:size(x, 2)
    %    theClass(i) = max(
    %end
end

