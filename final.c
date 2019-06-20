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

#define NODES1 300
#define NODES2 100
#define NODES3 10
#define PIXELS 28
#define LEARN 0.1
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
    double v[3][NODES1]; /* v matrix as input for activation function */
    double y[3][NODES1]; /* y matrix as output for activation function */
    double delta[3][NODES1]; /* local gradient/sensitivity matrix */
    double bias[3][NODES1]; /* bias matrix */
} config_t;

/* ---------------------------------------------------------------------- */
/* prototypes */

void help(void); /* print some help */
void copyr(void); /* print version and copyright information */

double activation(double v); /* funcao de ativacao */
double d_activation(double v); /* derivada da funcao de ativacao */
struct sconfig iniciarMapas(struct sconfig *c, struct s_header h); /* inicia os mapas de pesos e bias */
double * iniciarW(int a, int b, double mapa[a][b]); /* inicia as matrizes de weights baseado nas entradas */
double * normal (int inicio, unsigned char in[], double out[]); /* normaliza n imagens de entrada */
int train(void); /* treina uma rede neural */
void help(void); /* imprime ajuda */

int runtest(void);

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
                runtest();
                break;
            case '?':
            default:
                help();
                return EXIT_FAILURE;
        }

    return EXIT_SUCCESS;
}

/* funcao de ativacao */
double activation(double v)
{
    double y;
    double expvalue = exp(-v);
    y = 1/(1 + expvalue);

    return y;
}

/* derivada da funcao de ativacao */
double d_activation(double v)
{
    double dy;
    double expvalue = exp(v);
    dy = expvalue/(pow((expvalue + 1), 2));

    return dy;
}

double * iniciarW(int a, int b, double mapa[a][b])
{
    int i, j;

    for(i = 0; i < a; i++)
        for(j = 0; j < b; j++)
            {
                mapa[i][j] = (rand()%100/100.0) - 0.5;
            }
    
    printf("\n");
    return *mapa;
}

struct sconfig iniciarMapas(struct sconfig *c, struct s_header h)
{
    int i, j;

    /* Weights */
    iniciarW((int)NODES1, (int)h.lin*h.col, c -> wmap1);    
    iniciarW((int)NODES2, (int)NODES1, c -> wmap2);    
    iniciarW((int)NODES3, (int)NODES2, c -> wmap3);
    
    /* Bias */
    for(j = 0; j < 3; j++)
        for(i = 0; i < NODES1; i++)
            c -> bias[j][i] = (rand()%100/100.0) - 0.5;
    
    return *c;
}

double * normal (int inicio, unsigned char in[], double out[785])
{
    int i, j;
    for(i = inicio; i < inicio+1; i++)
        for(j = 0; j <= 784; j++)
        {
            if(j == 784)
                out[j] = in[i*785 + j]*1.0;
            else
                out[j] = ((in[i*785 + j]*1.0)/255);
        }    

    return out;
}

/* treina uma rede neural */
int train(void) 
{
    /* declaracoes de variaveis locais */
    header_t h;
    config_t *c = (config_t *)malloc(sizeof(config_t));
    int i, j, k, n;
    double erro[NODES3],
           saidaideal[NODES3]; /* vetor resultado ideal ou label do numero lido */
    double *imgVec;
    double *vin;
    double *sum;
    unsigned char *img;
    unsigned char *entradateste;
    header_t htest;
    FILE *fp;
    FILE *arquivomap;
    FILE *testep;

    /* codigo */
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

    /* alocação de memória */
    sum = (double *)malloc(NODES3 * sizeof(double));
    img = (unsigned char *)malloc(sizeof(unsigned char)*((h.lin*h.col)+1)*h.ni);
    imgVec = (double *)malloc(sizeof(double)*((h.lin*h.col)+1));

    i=0;
    while((n=fread(&img[i], sizeof(unsigned char), 1, fp)) == 1)
        i++;
    fclose(fp);
        
    printf("\n");

    /* inicializacao dos mapas de pesos e bias */
    iniciarMapas(c, h);    

    /* 'i': imagem atual -- numero total de imagens para treinar a rede */
    for(i = 0; i < h.ni; i++)
    {
        c -> eta = LEARN*h.ni/(i+h.ni); /* atualização na learning rate */

        /* normalizacao dos valores de entrada */
        normal(i, img, imgVec);

        /* . . . . . . . . . . . . . . */
        /* FORWARD COMPUTATION */
        //fowardComputation(c);

        /* primeiro layer */
        for(j = 0; j < NODES1; j++)
        {
            c -> v[1-1][j] = c -> bias[1-1][j];
            for(k = 0; k < 784; k++)
                c -> v[1-1][j] += c -> wmap1[j][k] * imgVec[k];

            c -> y[1-1][j] = activation(c -> v[1-1][j]);
        }

        /* segundo layer */
        for(j = 0; j < NODES2; j++)
        {
            c -> v[2-1][j] = c -> bias[2-1][j];
            for(k = 0; k < NODES1; k++)
                c -> v[2-1][j] += c -> wmap2[j][k] * c -> y[1-1][k];

            c -> y[2-1][j] = activation(c -> v[2-1][j]);
        }

        /* terceiro layer */
        for(j = 0; j < NODES3; j++)
        {
            c -> v[3-1][j] = c -> bias[3-1][j];
            for(k = 0; k < NODES2; k++)
                c -> v[3-1][j] += c -> wmap3[j][k] * c -> y[2-1][k];

            c -> y[3-1][j] = activation(c -> v[3-1][j]); /* a saida v3 eh o vetor resultado que nos diz o numero que a rede supoe que seja */
            if(DEBUG) printf("%1.2lf ", c -> y[3-1][j]);
        }
        if(DEBUG) printf("\n");

        /* . . . . . . . . . . . . . . */
        /* BACKWARD COMPUTATION */

        /* calculo do erro */
        for(j = 0; j < NODES3; j++)
            saidaideal[j] = 0;

        saidaideal[(int)imgVec[784]] = 1;
        for(j = 0; j < NODES3; j++)
            erro[j] = c -> y[3-1][j] - saidaideal[j];

        /* atualização das matrizes */
        
        /* delta do terceiro layer */
        for(j = 0; j < NODES3; j++)
            c -> delta[3-1][j] = (2)*erro[j] * d_activation(c -> v[3-1][j]);

        /* terceiro layer */
        for(j = 0; j < NODES3; j++)
        {
            for(k = 0; k < NODES2; k++)
            {
                c -> wmap3[j][k] -= (c -> eta * c -> delta[3-1][j] * c -> y[2-1][k]);
            }
            c -> bias[3-1][j] -= (c -> eta * c -> delta[3-1][j]);
        }

        /* delta do segundo layer */
        for(j = 0; j < NODES2; j++)
            c -> delta[2-1][j] = 0;

        for(j = 0; j < NODES3; j++)
        {
            for(k = 0; k < NODES2; k++)
                c -> delta[2-1][k] += c -> delta[3-1][k] * c -> wmap3[j][k];
        }
        
        for(j = 0; j < NODES2; j++)
            c -> delta[2-1][j] = c -> delta[2-1][j] * d_activation(c -> v[2-1][j]);

        /* segundo layer */
        for(j = 0; j < NODES2; j++)
        {
            for(k = 0; k < NODES1; k++)
                {
                    c -> wmap2[j][k] -= (c -> eta * c -> delta[2-1][j] * c -> y[1-1][k]);
                }
            c -> bias[2-1][j] -= (c -> eta * c -> delta[2-1][j]);
        }

        /* delta do primeiro layer */
        for(j = 0; j < NODES1; j++)
            c -> delta[1-1][j] = 0;

        for(j = 0; j < NODES2; j++)
        {
            for(k = 0; k < NODES1; k++)
                c -> delta[1-1][k] += c -> delta[2-1][k] * c -> wmap2[j][k];
        }
        
        for(j = 0; j < NODES1; j++)
            c -> delta[1-1][j] = c -> delta[1-1][j] * d_activation(c -> v[1-1][j]);

        /* primeiro layer */
        for(j = 0; j < NODES1; j++)
        {
            for(k = 0; k < 784; k++)
            {
                c -> wmap1[j][k] -= (c -> eta * c -> delta[1-1][j] * imgVec[k]);
            }
            c -> bias[1-1][j] -= (c -> eta * c -> delta[1-1][j]);
        }
    }

    /* . . . . . . . . . . . . . */
    /* teste da rede */

    if((testep=fopen("test-4k-images-labels", "rb"))==NULL)
    {
        printf("Nao consigo abrir arquivo %s\n", "test-4k-images-labels");
        exit(1);
    }

    fread(&htest, sizeof(header_t), 1, testep);
    entradateste = (unsigned char *)malloc(785 * sizeof(unsigned char));
    vin = (double *)malloc(785 * sizeof(double));
    erro[0] = 0;
    for(i=0; i<100; i++)
    {
        if((n=fread(entradateste, sizeof(unsigned char), 785, testep)) == 785)
        {
            /* normalizacao dos valores de entrada */
            normal(0, entradateste, vin);

            /* . . . . . . . . . . . . . . */
            /* FORWARD COMPUTATION */

            /* primeiro layer */
            for(j = 0; j < NODES1; j++)
            {
                c -> v[1-1][j] = c -> bias[1-1][j];
                for(k = 0; k < 784; k++)
                    c -> v[1-1][j] += c -> wmap1[j][k] * vin[k];

                c -> y[1-1][j] = activation(c -> v[1-1][j]);
            }

            /* segundo layer */
            for(j = 0; j < NODES2; j++)
            {
                c -> v[2-1][j] = c -> bias[2-1][j];
                for(k = 0; k < NODES1; k++)
                    c -> v[2-1][j] += c -> wmap2[j][k] * c -> y[1-1][k];

                c -> y[2-1][j] = activation(c -> v[2-1][j]);
            }

            /* terceiro layer */
            for(j = 0; j < NODES3; j++)
            {
                c -> v[3-1][j] = c -> bias[3-1][j];
                for(k = 0; k < NODES2; k++)
                    c -> v[3-1][j] += c -> wmap3[j][k] * c -> y[2-1][k];

                c -> y[3-1][j] = activation(c -> v[3-1][j]);
            }
            
            /* Verificacao */            
            for(j = 0; j < NODES3; j++)
            {
                sum[j] = 0;
                for(k = 0; k < NODES3; k++)
                {   
                    if(k == j)
                        sum[j] += pow((c -> y[3-1][k]-1),2) ;
                    else
                        sum[j] += pow(c -> y[3-1][k],2) ;
                }
            }
            
            k = 0;
            for(j = 1; j < NODES3; j++)
                if(sum[k] > sum[j])
                    k = j;
           
            /* 
            k = 0;
            for(j = 1; j < NODES3; j++)
                if(c -> y[3-1][j] > c -> y[3-1][k])
                    k = j;
            */
            printf("%u - ", entradateste[784]);
            printf("%d  ", k);
            if(entradateste[784] == k)
                printf("\n");
            else
            {
                printf("X\n");
                erro[0] += 1;
            }           
        }
    }
    erro[1] = (100.0 * erro[0])/i;
    printf("\nErro: %.2f%%\n", erro[1]);

    free(imgVec);
    free(img);
    free(sum);
    free(entradateste);
    fclose(testep);
    printf("\nErro: %.2f%%\n", erro[1]);
    /* . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . */
    /* save the neural network as a binary file */
    /* salvar o mapa gerado em dados binarios */
    arquivomap = fopen("wmap", "wb");
    if(arquivomap == NULL)
    {
        printf("Erro abrindo arquivo binario.\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        for(i = 0; i < NODES1; i++)
            fwrite(&c -> wmap1[i], 784*sizeof(double), 1, arquivomap);

        for(i = 0; i < NODES2; i++)
            fwrite(&c -> wmap2[i], NODES1*sizeof(double), 1, arquivomap);

        for(i = 0; i < NODES3; i++)
            fwrite(&c -> wmap3[i], NODES2*sizeof(double), 1, arquivomap);
    }

    free(c);
    fclose(arquivomap); 
}


int runtest(void)
{
    printf("funcao nao implementada\n");
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

