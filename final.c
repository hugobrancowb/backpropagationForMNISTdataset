/* vamos ler 6000 imagens */
/* hexdump */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
/* #include <unistd.h> */
/* #include <error.h> */
/* #include <errno.h> */
#include <time.h>
#include <string.h>

/* ---------------------------------------------------------------------- */
/* definitions */

#ifndef VERSION /* gcc -DVERSION="0.1.160612.142306" */
#define VERSION "20190416.195702" /* Version Number (string) */
#endif

/* Debug */
#ifndef DEBUG /* gcc -DDEBUG=1 */
#define DEBUG 0 /* Activate/deactivate debug mode */
#endif

#if DEBUG==0
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
    int ni; /* test = 10000 */
    int lin; /* rows = 28 */
    int col; /* columns = 28 */
} header_t;

typedef struct sconfig
{
    double eta; /* eta: learning rate */
    int counter; /* interation counter */

    double v1[NODES1];
    double v2[NODES2];
    double v3[NODES3];
    
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

/* ---------------------------------------------------------------------- */
/* types */
/* 
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  10000            number of images 
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

int main(void)
{
    /* declaracoes de variaveis locais */
    header_t h;
    config_t c;
    kohonen_t koh;
    FILE *fp;
    double v,
           bias = 1;
    int indice;
    int i, j, k, n;

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

    unsigned char imgCHAR[h.ni][(h.lin*h.col)+1]; /* linha: imagens // coluna: 784 pixels + 1 label */
    double img[h.ni][(h.lin*h.col)+1];
    /* para vetores maiores, usar malloc e alocacao dinamica */
    /* unsigned char *img; */ /* ponteiro para matriz de dados */
    /* img = (unsigned char *)malloc(sizeof(unsigned char)*h.lin*h.col*h.ni); */

    i=0;
    while((n=fread(&imgCHAR[i], sizeof(kohonen_t), 1, fp)) == 1)
        i++;
    
    /* normalizacao dos valores de entrada*/
    for(i = 0; i < h.ni; i++)
        for(j = 0; j < 784; j++)
            img[i][j] = imgCHAR[i][j]/255;
    
    /* inicializacao dos mapas de pesos */
    for(i = 0; i < NODES1; i++)
        for(j = 0; j < h.lin*h.col; j++)
            c.wmap1[i][j] = (rand()%100)/100000.0;

    for(i = 0; i < NODES2; i++)
        for(j = 0; j < NODES1; j++)
            c.wmap2[i][j] = (rand()%100)/100000.0;

    for(i = 0; i < NODES3; i++)
        for(j = 0; j < NODES2; j++)
            c.wmap3[i][j] = (rand()%100)/100000.0;

    /* 'i': imagem atual -- numero total de imagens para treinar a rede */
    for(i = 0; i < h.ni; i++)
    {
        /* . . . . . . . . . . . . . . */
        /* FORWARD COMPUTATION */
        /* primeiro layer */
        for(j = 0; j < NODES1; j++)
        {
            c.v1[j] = bias;
            for(k = 0; k < 784; k++)
            {
                c.v1[j] += c.wmap1[j][k] * img[i][k];
            }
            c.v1[j] = activation(c.v1[j]);
        }

        /* . . . . . . . . . . . . . . */
        /* BACKWARD COMPUTATION */
    }

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


/* ---------------------------------------------------------------------- */
/* Prints help information 
 *  usually called by opt -h or --help
 */
void help(void)
{
    IFDEBUG("help()");
    printf("%s - %s\n", "lebase", "le base de digitos de 0 a 9");
    printf("\nUsage: %s\n\n", "lebase");
    printf("This program does...\n");
    /* add more stuff here */
    printf("\nExit status:\n\t0 if ok.\n\t1 some error occurred.\n");
    printf("\nTodo:\n\tLong options not implemented yet.\n");
    printf("\nAuthor:\n\tWritten by %s <%s>\n\n", "Ruben Carlo Benante", "rcb@beco.cc");
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

