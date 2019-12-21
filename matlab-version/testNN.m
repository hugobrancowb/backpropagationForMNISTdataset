function [precisao] = testNN(W1,W2,W3,B1,B2,B3,input)

acertos = 0;
randomvector = randperm(size(input,1));
input = input(randomvector,:);

c1 = 300;
c2 = 100;
c3 = 10;

for n = 1:size(input,1)
    Y0 = input(n,2:end)';
    Y0 = Y0/255;
    labelCorreto = input(n,1);
    
    % Foward Phase
    V1 = (W1 * Y0) + B1;
    Y1 = Sigmoid(V1);
    
    V2 = (W2 * Y1) + B2;
    Y2 = Sigmoid(V2);
    
    V3 = (W3 * Y2) + B3;
    Y3 = Sigmoid(V3);
    
    % Resultado
    [~,output] = max(Softmax(Y3));
    output = output - 1;
    
    acertos = acertos + (output == labelCorreto);
end
    precisao = acertos/size(input,1);
    fprintf("Precisão: %.1f\n", precisao*100);
end

