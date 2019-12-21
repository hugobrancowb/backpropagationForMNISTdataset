function [W1,W2,W3,B1,B2,B3] = Backprop(input, alfa)

c1 = 300;
c2 = 100;
c3 = 10;

% Matrizes de Pesos
W1 = rand(c1,784)-.5;
W2 = rand(c2,c1)-.5;
W3 = rand(c3,c2)-.5;

B1 = rand(c1,1)-.5;
B2 = rand(c2,1)-.5;
B3 = rand(c3,1)-.5;

r = randperm(size(input,1));
input = input(r,:);

for n = 1:size(input,1)

    Y0 = input(n,2:end)';
    Y0 = Y0/255;
    
    % Foward Phase
    V1 = (W1 * Y0) + B1;
    Y1 = Sigmoid(V1);
    
    V2 = (W2 * Y1) + B2;
    Y2 = Sigmoid(V2);
    
    V3 = (W3 * Y2) + B3;
    Y3 = Sigmoid(V3);
    
    % Resultado
    R = zeros(c3,1);
    R( input(n,1) + 1 ) = 1;
    
    % Backward Phase
    E3 = (R - Y3);
    
    Delta3 = -2*dSig(V3)*E3;
    dW3 = alfa * Delta3 * Y2';
    W3 = W3 - dW3;
    B3 = B3 - alfa * Delta3;
    
    Delta2 = dSig(V2)*W3'*Delta3;
    dW2 = alfa * Delta2 * Y1';
    W2 = W2 - dW2;
    B2 = B2 - alfa * Delta2;
    
    Delta1 = dSig(V1)*W2'*Delta2;    
    dW1 = alfa * Delta1 * Y0';    
    W1 = W1 - dW1;
    B1 = B1 - alfa * Delta1;

end

end

