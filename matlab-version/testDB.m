function [saida] = testDB(amostra)

    total = 10000;
    if(amostra>total),amostra=total;end
    
    arq = fopen('t10k-images.idx3-ubyte', 'r', 'b');
    lbl = fopen('t10k-labels.idx1-ubyte', 'r', 'b');
    
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

