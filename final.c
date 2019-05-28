/* vamos ler 6000 imagens */
/* hexdump */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
/* #include <unistd.h> */
/* #include <error.h> */
/* #include <errno.h> */
#include <getopt.h> /* get options from system argc/argv */
#include <time.h>
#include <string.h>

/* ---------------------------------------------------------------------- */
/* definitions */

#ifndef VERSION /* gcc -DVERSION="0.1.160612.142306" */
#define VERSION "20190416.195702" /* Version Number (string) */
#endif

/* Debug */
#ifndef DEBUG /* gcc -DDEBUG=1 */
#define DEBUG 1 /* Activate/deactivate debug mode */
#endif

#if DEBUG==1
#define NDEBUG
#endif
/* #include <assert.h> */ /* Verify assumptions with assert. Turn off with #define NDEBUG */ 

/* Debug message if DEBUG on */
#define IFDEBUG(M) if(DEBUG) fprintf(stderr, "[DEBUG file:%s line:%d]: " M "\n", __FILE__, __LINE__); else {;}

/* limits */
#define SBUFF 256 /* string buffer */
/* #define NOMEARQ "t10k-images-idx3-ubyte" */
#define NOMEARQ "train-6k-images-labels"

#define NODES1 16
#define NODES2 16
#define NODES3 10
#define PIXELS 28
#define N_IMAGENS 6000

typedef struct s_header
{
    int magic; /* magic number 2051 */
    int ni; /* test = 6000 */
    int lin; /* rows = 28 */
    int col; /* columns = 28 */
} header_t;

typedef struct sconfig
{
    double eta; /* eta: learning rate */
    int counter; /* interation counter */
    
    double wmap1[NODES1][784]; /* first neuron layer - hidden */
    double wmap2[NODES2][NODES1]; /* second neuron layer - hidden */
    double wmap3[NODES3][NODES2]; /* third neuron layer - output */
} config_t;

typedef struct skohonen
{
    unsigned char entrada[785];
} kohonen_t;

/* ---------------------------------------------------------------------- */
/* prototypes */

void help(void); /* print some help */
void copyr(void); /* print version and copyright information */

double activation(double v); /* funcao de ativacao */
double d_activation(double v); /* derivada da funcao de ativacao */
int train(void); /* treina uma rede neural */
void help(void); /* imprime ajuda */

/* ---------------------------------------------------------------------- */
/* types */
/* 
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  6000             number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
.... 
xxxx     unsigned byte   ??               pixel 
xxxx     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               pixel
xxxx     unsigned byte   ??               label
*/

int main(int argc, char *argv[])
{
    int opt; /* return from getopt() */

    IFDEBUG("Starting optarg loop...");

    /* getopt() configured options:
     *        -h  help
     *        -t  train a neural network
     *        -r  run the saved neural network
     */

    opterr = 0;
    while((opt = getopt(argc, argv, "trh")) != EOF)
        switch(opt)
        {
            case 't':
                train();
                break;
            case 'r':
                /*runtest();*/
                break;
            case '?':
            default:
                help();
                return EXIT_FAILURE;
        }

    help();
    return EXIT_SUCCESS;
}

/* funcao de ativacao */
double activation(double v)
{
    double y = 1/(1+exp(-v));
    return y;
}

/* derivada da funcao de ativacao */
double d_activation(double v)
{
    double av = activation(v);
    double dy = av * (1 - av);
    return dy;
}

/* treina uma rede neural */
int train(void) {
    /* declaracoes de variaveis locais */
    header_t h;
    /* config_t c; */
    struct sconfig *c = (struct sconfig*)malloc(sizeof(struct sconfig));
    /* kohonen_t koh; */
    FILE *fp;
    double erro[NODES3],
           saidaideal[NODES3], /* vetor resultado ideal ou label do numero lido */
           bias;
    int i, j, k, n;

    /* codigo */
    c -> eta = 0.005;
    bias = 1;

    srand(time(NULL));

    if((fp=fopen(NOMEARQ, "rb"))==NULL)
    {
        printf("Nao consigo abrir arquivo %s\n", NOMEARQ);
        exit(1);
    }

    fread(&h, sizeof(header_t), 1, fp);

    printf("Teste de leitura:\n");
    printf("Num. magico: %d\n", h.magic);
    printf("Num. imagens: %d\n", h.ni);
    printf("Num. linhas por imagem: %d\n", h.lin);
    printf("Num. colunas por imagem: %d\n", h.col);
    printf("Lendo imagens %d x %d\n", h.lin, h.col);
    
    unsigned char *img;
    img = (unsigned char *)malloc(sizeof(unsigned char)*((h.lin*h.col)+1)*h.ni);

    i=0;
    while((n=fread(&img[i], sizeof(unsigned char), 1, fp)) == 1)
        i++;
    fclose(fp);
        
    printf("\n");

    double *imgVec;
    imgVec = (double *)malloc(sizeof(double)*((h.lin*h.col)+1)*h.ni);

    /* normalizacao dos valores de entrada */
    for(i = 0; i < h.ni; i++)
        for(j = 0; j <= 784; j++)
        {
            if(j == 784)
                imgVec[i*785 + j] = img[i*785 + j]*1.0;
            else
                imgVec[i*785 + j] = (img[i*785 + j]*1.0);
        }

    for(i = 0; i < h.ni; i++)
        for(j = 0; j < 784; j++)
                imgVec[i*785 + j] = imgVec[i*785 + j]/255 - 0.5;

    /* variaveis */
    double *v1;
    double *v2;
    double *v3;
    v1 = (double *)malloc(NODES1*sizeof(double));
    v2 = (double *)malloc(NODES2*sizeof(double));
    v3 = (double *)malloc(NODES3*sizeof(double));
    double *y1;
    double *y2;
    double *y3;
    y1 = (double *)malloc(NODES1*sizeof(double));
    y2 = (double *)malloc(NODES2*sizeof(double));
    y3 = (double *)malloc(NODES3*sizeof(double));
    double *delta1;
    double *delta2;
    double *delta3;
    delta1 = (double *)malloc(NODES1*sizeof(double));
    delta2 = (double *)malloc(NODES2*sizeof(double));
    delta3 = (double *)malloc(NODES3*sizeof(double));

    /* inicializacao dos mapas de pesos */
    for(i = 0; i < NODES1; i++)
        for(j = 0; j < h.lin*h.col; j++)
            c -> wmap1[i][j] = (rand()%100 - rand()%50)/100.0;

    for(i = 0; i < NODES2; i++)
        for(j = 0; j < NODES1; j++)
            c -> wmap2[i][j] = (rand()%100 - rand()%50)/100.0;

    for(i = 0; i < NODES3; i++)
        for(j = 0; j < NODES2; j++)
            c -> wmap3[i][j] = (rand()%100 - rand()%50)/100.0;

    /* 'i': imagem atual -- numero total de imagens para treinar a rede */
    for(i = 0; i < h.ni; i++)
    {
        /* . . . . . . . . . . . . . . */
        /* FORWARD COMPUTATION */

        /* primeiro layer */
        for(j = 0; j < NODES1; j++)
        {
            v1[j] = bias;
            for(k = 0; k < 784; k++)
            {
                v1[j] += c -> wmap1[j][k] * imgVec[i*785 + k];
            }
            y1[j] = activation(v1[j]);
        }

        /* segundo layer */
        for(j = 0; j < NODES2; j++)
        {
            v2[j] = bias;
            for(k = 0; k < NODES1; k++)
            {
                v2[j] += c -> wmap2[j][k] * y1[k];
            }
            y2[j] = activation(v2[j]);
        }

        /* terceiro layer */
        for(j = 0; j < NODES3; j++)
        {
            v3[j] = bias;
            for(k = 0; k < NODES2; k++)
            {
                v3[j] += c -> wmap3[j][k] * y2[k];
            }
            y3[j] = activation(v3[j]); /* a saida v3 eh o vetor resultado que nos diz o numero que a rede supoe que seja */

            saidaideal[j] = 0;
            erro[j] = 0;
        }

        /* . . . . . . . . . . . . . . */
        /* BACKWARD COMPUTATION */

        /* calculo do erro */
        saidaideal[(int)imgVec[i*785 + 784]] = 1;
        for(j = 0; j < NODES3; j++)
            erro[j] = saidaideal[j] - y3[j];
        
        /* terceiro layer */
        for(j = 0; j < NODES3; j++)
        {
            delta3[j] = erro[j] * d_activation(v3[j]);
            for(k = 0; k < NODES2; k++)
                c -> wmap3[j][k] += c -> eta * delta3[j] * y2[k];
        }

        /* segundo layer */
        for(j = 0; j < NODES2; j++)
        {
            delta2[j] = 0;
            for(k = 0; k < NODES3; k++)
                delta2[j] += delta3[k] * c -> wmap3[k][j];
            
            delta2[j] = delta2[j] * d_activation(v2[j]);

            for(k = 0; k < NODES1; k++)
                c -> wmap2[j][k] += c -> eta * delta2[j] * y1[k];
        }

        /* primeiro layer */
        for(j = 0; j < NODES1; j++)
        {
            delta1[j] = 0;
            for(k = 0; k < NODES2; k++)
                delta1[j] += delta2[k] * c -> wmap2[k][j];
            
            delta1[j] = delta1[j] * d_activation(v1[j]);

            for(k = 0; k < 784; k++)
                c -> wmap1[j][k] += c -> eta * delta1[j] * imgVec[i*785 + k];
        }
    }

    /* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */
    /* teste da rede */

    header_t htest;
    FILE *testep;
    if((testep=fopen("test-4k-images-labels", "rb"))==NULL)
    {
        printf("Nao consigo abrir arquivo %s\n", "test-4k-images-labels");
        exit(1);
    }

    fread(&htest, sizeof(header_t), 1, testep);
    unsigned char *entradateste;
    entradateste = (unsigned char *)malloc(785 * sizeof(unsigned char));
    double *vin;
    vin = (double *)malloc(785 * sizeof(double));
    for(i=0; i<10; i++)
    {
        if((n=fread(entradateste, sizeof(unsigned char), 785, testep)) == 785)
        {
            for(j = 0; j <= 784; j++)
            {
                if(j == 784)
                    vin[j] = entradateste[j]*1.0;
                else
                    vin[j] = entradateste[j]/255.0;
            }
            
            /* . . . . . . . . . . . . . . */
            /* FORWARD COMPUTATION */

            /* primeiro layer */
            for(j = 0; j < NODES1; j++)
            {
                v1[j] = bias;
                for(k = 0; k < 784; k++)
                {
                    v1[j] += c -> wmap1[j][k] * entradateste[k];
                }
                y1[j] = activation(v1[j]);
            }

            /* segundo layer */
            for(j = 0; j < NODES2; j++)
            {
                v2[j] = bias;
                for(k = 0; k < NODES1; k++)
                {
                    v2[j] += c -> wmap2[j][k] * y1[k];
                }
                y2[j] = activation(v2[j]);
            }

            double max=0;
            int indice=0;
            /* terceiro layer */
            for(j = 0; j < NODES3; j++)
            {
                v3[j] = bias;
                for(k = 0; k < NODES2; k++)
                {
                    v3[j] += c -> wmap3[j][k] * y2[k];
                }
                y3[j] = activation(v3[j]); /* a saida v3 eh o vetor resultado que nos diz o numero que a rede supoe que seja */
                
                if(y3[j] > max)
                {
                    max = y3[j];
                    indice = j;
                }
                printf("%1.3lf  ",y3[j]);
            }
            printf("\nnumero lido // adivinhado:\t%u\t",entradateste[784]);
            printf("%d\n",indice);
        }
    }

    free(entradateste); fclose(testep);
    /* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */
    /* save the neural network as a binary file */
    /* salvar o mapa gerado em dados binarios */
    FILE *temp;
    if((temp=fopen("wmap", "wb"))!=NULL)
    {
        for(i = 0; i < NODES1; i++)
            fwrite(&c -> wmap1[i], 784*sizeof(double), 1, temp);

        for(i = 0; i < NODES2; i++)
            fwrite(&c -> wmap2[i], NODES1*sizeof(double), 1, temp);

        for(i = 0; i < NODES3; i++)
            fwrite(&c -> wmap3[i], NODES2*sizeof(double), 1, temp);

        fclose(temp); 
    }

    free(imgVec); free(c);
    free(v1); free(v2); free(v3); 
    free(y1); free(y2); free(y3); 
    free(delta1); free(delta2); free(delta3);

    exit(EXIT_SUCCESS);
}


/* ---------------------------------------------------------------------- */
/* Prints help information 
 *  usually called by opt -h or --help
 */
void help(void)
{    
    printf("-t\t train the neural network\n");
    printf("-r\t run the neural network\n");
    printf("-h\t help\n\n");
    /* printf("\nExit status:\n\t0 if ok.\n\t1 some error occurred.\n"); */
    /* printf("\nTodo:\n\tLong options not implemented yet.\n"); */
    printf("\nAuthor:\n\tWritten by %s <%s>\n\n", "Hugo Branco W. Barbosa", "hugobrancowb@gmail.com");
    return;
}

/* ---------------------------------------------------------------------- */
/* Prints version and copyright information 
 *  usually called by opt -V
 */
void copyr(void)
{
    IFDEBUG("copyr()");
    printf("%s - Version %s\n", "lebase", VERSION);
    printf("\nCopyright (C) %d %s <%s>, GNU GPL version 2 <http://gnu.org/licenses/gpl.html>. This  is  free  software: you are free to change and redistribute it. There is NO WARRANTY, to the extent permitted by law. USE IT AS IT IS. The author takes no responsability to any damage this software may inflige in your data.\n\n", 2019, "Ruben Carlo Benante", "rcb@beco.cc");
    return;
}

/* ---------------------------------------------------------------------- */
/* vi: set ai et ts=4 sw=4 tw=0 wm=0 fo=croql : C config for Vim modeline */
/* Template by Dr. Beco <rcb at beco dot cc> Version 20160612.142044      */

