% Matriz = reading(n);
%
% Abre a base de imagens e as armazena numa matriz junto com seus labels.
%
% Entrada
%     n    Número de imagens que queremos ler
%
% Saída
%     M    Matriz (n,785) onde cada linha é uma imagem. O primeiro elemento
%          de cada linha equivale ao label enquanto os demais elementos são
%          os valores de cada pixel.

function [saida] = trainDB(amostra)
    
    total = 60000;
    if(amostra>total),amostra=total;end
    
    arq = fopen('train-images.idx3-ubyte', 'r', 'b');
    lbl = fopen('train-labels.idx1-ubyte', 'r', 'b');
    
    [~] = fread(arq, 4, 'int32');
    [~] = fread(lbl, 2, 'int32');

    imagensdb = fread(arq,[784,total],'uint8');
    imagensdb = imagensdb';
    labelsdb = fread(lbl,[total,1],'uint8');
    imagensdb = [labelsdb,imagensdb];

    r = randperm(size(imagensdb, 1));
    imagensdb = imagensdb(r,:);

    saida = imagensdb(1:amostra,:);
    
    fclose(arq);
    fclose(lbl);

end