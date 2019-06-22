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
    double y[4][785]; /* y matrix as output for activation function */
    double delta[3][NODES1]; /* local gradient/sensitivity matrix */
    double bias[3][NODES1]; /* bias matrix */
} config_t;

/* ---------------------------------------------------------------------- */
/* prototypes */

void help(void); /* print some help */
void copyr(void); /* print version and copyright information */

double activation(double v); /* funcao de ativacao */
double d_activation(double v); /* derivada da funcao de ativacao */
struct sconfig iniciarMapas(struct sconfig *c); /* inicia os mapas de pesos e bias */
double * iniciarW(int a, int b, double mapa[a][b]); /* inicia as matrizes de weights baseado nas entradas */
struct sconfig normal (struct sconfig *c, int inicio, unsigned char in[]); /* normaliza n imagens de entrada */
struct sconfig fowardComputation(struct sconfig *c); /* executa os calculos das matrizes para o sentido direto da rede */
struct sconfig backwardComputation(struct sconfig *c); /* executa os calculos das matrizes para o sentido inverso da rede e atualiza suas matrizes */
struct sconfig signalFlow(struct sconfig *c, int a, int nodesPrev, int nodes, double wmap[nodes][nodesPrev]); /* realiza os cálculos da fowardComputation para cada camada. fowardComputation apenas chama signnalFlow três vezes, uam vez para cada camada de perceptrons */
struct sconfig backWeights(struct sconfig *c, int a, int nodesPrev, int nodes); /* atualiza as matrizes de pesos baseado nos novos valores de delta */
struct sconfig deltaBack(struct sconfig *c, int a, int nodesPrev, int nodes, double wmap[nodes][nodesPrev]); /* calcula os novos vetores delta baseado no erro calculado */
FILE mapaBias(struct sconfig *c, int opt, FILE *arquivo, int a, int size1, int size2, double mapa[size2][size1]); /* salva ou abre um arquivo contendo as matrizes de pesos e vetores bias */
int train(double lr); /* treina uma rede neural */
void help(void); /* imprime ajuda */

double runtest(int n);

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
    int n=1;
    double lr, erro;

    IFDEBUG("Starting optarg loop...");

    /* getopt() configured options:
     *        -h  help
     *        -t  train a neural network
     *        -r  run the saved neural network
     */

    opterr = 0;
    while((opt = getopt(argc, argv, "trgh")) != EOF)
        switch(opt)
        {
            case 't':
                n = train(LEARN);

                while(n > 0)
                {
                    printf("Deseja testar a rede? Digite o tamanho da amostra ou digite '0' para sair: ");
                    scanf("%d", &n);
                    
                    if(n)
                        runtest(n);
                }

                break;
            case 'r':
                while(n > 0)
                {
                    printf("Digite o tamanho da amostra ou digite '0' para sair: ");
                    scanf("%d", &n);
                    
                    if(n)
                        runtest(n);
                }
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
                mapa[i][j] = (rand()%100/100.0) - 0.5;

    return *mapa;
}

struct sconfig iniciarMapas(struct sconfig *c)
{
    int i, j, limite;

    /* Weights */
    iniciarW((int)NODES1, 784, c -> wmap1);    
    iniciarW((int)NODES2, (int)NODES1, c -> wmap2);    
    iniciarW((int)NODES3, (int)NODES2, c -> wmap3);
    
    /* Bias */
    for(j = 0; j < 3; j++)
    {
        switch (j)
        {
            case 0:
                limite = NODES1;
                break;
            case 1:
                limite = NODES2;
                break;
            case 2:
                limite = NODES3;
                break;
            
            default:
                printf("Ocorreu um erro inesperado.\n");
                exit(EXIT_FAILURE);
                break;
        }

        for(i = 0; i < limite; i++)
            c -> bias[j][i] = (rand()%100/100.0) - 0.5;
    }
    
    return *c;
}

struct sconfig normal (struct sconfig *c, int inicio, unsigned char in[])
{
    int j;

    for(j = 0; j <= 784; j++)
    {
        if(j != 784)
            c -> y[0][j] = ((in[inicio*785 + j]*1.0)/255);
        else
            c -> y[0][j] = in[inicio*785 + j]*1.0;
    }    

    return *c;
}

struct sconfig signalFlow(struct sconfig *c, int a, int nodesPrev, int nodes, double wmap[nodes][nodesPrev])
{
    int i, j;

    for(i = 0; i < nodes; i++)
    {
        c -> v[a][i] = c -> bias[a][i];
        for(j = 0; j < nodesPrev; j++)
            c -> v[a][i] += wmap[i][j] * c -> y[a][j];

        c -> y[a+1][i] = activation(c -> v[a][i]);
    }

    return *c;
}

struct sconfig backWeights(struct sconfig *c, int a, int nodesPrev, int nodes)
{
    int i, j;
    double value;
    
    for(i = 0; i < nodes; i++)
    {
        for(j = 0; j < nodesPrev; j++)
        {
            value = (c -> eta * c -> delta[a][i] * c -> y[a][j]);

            switch (a)
            {
                case 2:
                    c -> wmap3[i][j] -= value;
                    break;
                case 1:
                    c -> wmap2[i][j] -= value;
                    break;
                case 0:
                    c -> wmap1[i][j] -= value;
                    break;
                
                default:
                    printf("Ocorreu um erro inesperado.\n");
                    exit(EXIT_FAILURE);
                    break;
            }         
        }

        c -> bias[a][i] -= (c -> eta * c -> delta[a][i]);
    }

    return *c;
}

struct sconfig fowardComputation(struct sconfig *c)
{
    signalFlow(c, 0, 784, NODES1, c -> wmap1);
    signalFlow(c, 1, NODES1, NODES2, c -> wmap2);
    signalFlow(c, 2, NODES2, NODES3, c -> wmap3);

    return *c;
}

struct sconfig deltaBack(struct sconfig *c, int a, int nodesPrev, int nodes, double wmap[nodes][nodesPrev])
{
    int i, j;
    
    for(i = 0; i < nodes; i++)
    {
        for(j = 0; j < nodesPrev; j++)
        {
            if(i == 0)
                c -> delta[a][j] = c -> delta[a+1][i] * wmap[i][j];
            else
                c -> delta[a][j] += c -> delta[a+1][i] * wmap[i][j];   
            
            if(i == nodes - 1)
                c -> delta[a][j] = c -> delta[a][j] * d_activation(c -> v[a][j]);
        }
    }

    return *c;
}

struct sconfig backwardComputation(struct sconfig *c)
{
    int j;
    double erro[NODES3],
           saidaideal[NODES3]; /* vetor resultado ideal ou label do numero lido */
    
    /* calculo do erro */
    for(j = 0; j < NODES3; j++)
    {
        if(j == c -> y[0][784])
            saidaideal[j] = 1; 
        else
            saidaideal[j] = 0; 

        erro[j] = c -> y[3][j] - saidaideal[j];
    }

    /* atualização das matrizes */

    /* delta do terceiro layer */
    for(j = 0; j < NODES3; j++)
        c -> delta[2][j] = (2)*erro[j] * d_activation(c -> v[2][j]);
        
    /* peso do terceiro layer */
    backWeights(c, 2, NODES2, NODES3);
    
    /* delta e peso do segundo layer */
    deltaBack(c, 1, NODES2, NODES3, c -> wmap3);
    backWeights(c, 1, NODES1, NODES2);

    /* delta e peso do primeiro layer */
    deltaBack(c, 0, NODES1, NODES2, c -> wmap2);
    backWeights(c, 0, 784, NODES1);

    return *c;
}

/* treina uma rede neural */
int train(double lr) 
{
    /* declaracoes de variaveis locais */
    header_t h;
    config_t *c = (config_t *)malloc(sizeof(config_t));
    int i, n;
    unsigned char *img;
    FILE *fp;
    FILE *arquivomap;

    /* codigo */
    srand(time(NULL));

    if((fp=fopen(NOMEARQ, "rb"))==NULL)
    {
        printf("Nao consigo abrir arquivo %s\n", NOMEARQ);
        exit(1);
    }

    fread(&h, sizeof(header_t), 1, fp);

    /* printf("Teste de leitura:\n");
    printf("Num. magico: %d\n", h.magic);
    printf("Num. imagens: %d\n", h.ni);
    printf("Num. linhas por imagem: %d\n", h.lin);
    printf("Num. colunas por imagem: %d\n", h.col);
    printf("Lendo imagens %d x %d\n", h.lin, h.col); */

    /* alocação de memória */
    img = (unsigned char *)malloc(sizeof(unsigned char)*((h.lin*h.col)+1)*h.ni);

    i=0;
    while((n=fread(&img[i], sizeof(unsigned char), 1, fp)) == 1)
        i++;
    fclose(fp);
        
    /* printf("\n"); */

    /* inicializacao dos mapas de pesos e bias */
    iniciarMapas(c);
    c -> eta = lr;    

    printf("Construindo a rede neural...\n");
    /* 'i': imagem atual -- numero total de imagens para treinar a rede */
    for(i = 0; i < h.ni; i++)
    {
        /* c -> eta = LEARN*h.ni/(i+h.ni+1); */ /* atualização na learning rate */

        /* normalizacao dos valores de entrada */
        normal(c, i, img);

        /* FORWARD COMPUTATION */
        fowardComputation(c);

        /* BACKWARD COMPUTATION */
        backwardComputation(c);
    }
    printf("Rede construida!\n");    

    free(img);
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
        mapaBias(c, 1, arquivomap, 0, 784, NODES1, c -> wmap1);
        mapaBias(c, 1, arquivomap, 1, NODES1, NODES2, c -> wmap2);
        mapaBias(c, 1, arquivomap, 2, NODES2, NODES3, c -> wmap3);
    }

    free(c);
    fclose(arquivomap); 

    return 1;
}

FILE mapaBias(struct sconfig *c, int opt, FILE *arquivo, int a, int size1, int size2, double mapa[size2][size1])
{
    /* opt = 1: salva arquivo */
    /* opt = 2: abre arquivo */

    int i;

    if(opt)
    {
        for(i = 0; i < size2; i++)
            fwrite(&mapa[i], size1*sizeof(double), 1, arquivo);

        for(i = 0; i < size2; i++)
            fwrite(&c-> bias[a][i], sizeof(double), 1, arquivo);
    }
    else
    {
        
        for(i = 0; i < size2; i++)
            fread(&mapa[i], size1*sizeof(double), 1, arquivo);

        for(i = 0; i < size2; i++)
            fread(&c-> bias[a][i], sizeof(double), 1, arquivo);
    }    
    
    return *arquivo;
}

double runtest(int n)
{
    int i, j, k, init, counter;
    double *sum;
    unsigned char *entradateste;
    header_t h;
    config_t *c = (config_t *)malloc(sizeof(config_t));
    FILE *arquivomap;
    FILE *testefile;

    sum = (double *)malloc(NODES3 * sizeof(double));

    srand(time(NULL));

    /* abrir o mapa gerado */
    arquivomap = fopen("wmap", "rb");
    if(arquivomap == NULL)
    {
        printf("Erro abrindo arquivo binario.\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        mapaBias(c, 0, arquivomap, 0, 784, NODES1, c -> wmap1);
        mapaBias(c, 0, arquivomap, 1, NODES1, NODES2, c -> wmap2);
        mapaBias(c, 0, arquivomap, 2, NODES2, NODES3, c -> wmap3);
    }

    if((testefile=fopen("test-4k-images-labels", "rb"))==NULL)
    {
        printf("Nao consigo abrir arquivo %s\n", "test-4k-images-labels");
        exit(1);
    }

    fread(&h, sizeof(header_t), 1, testefile);
    entradateste = (unsigned char *)malloc(785 * sizeof(unsigned char));
    double erros = 0;

    init = rand()%4000 + 1;

    while(init > 4000-n )
        init = rand()%4000 + 1;

    for(i=0; i < init; i++)
        fread(entradateste, sizeof(unsigned char), 785, testefile);

    counter = 1;
    for(i=init; i < (init + n); i++)
    {
        if((n=fread(entradateste, sizeof(unsigned char), 785, testefile)) == 785)
        {
            /* normalizacao dos valores de entrada */
            normal(c, 0, entradateste);
            
            /* FORWARD COMPUTATION */
            fowardComputation(c);
            
            /* Verificacao */            
            for(j = 0; j < NODES3; j++)
            {
                sum[j] = 0;
                for(k = 0; k < NODES3; k++)
                {   
                    if(k == j)
                        sum[j] += pow((c -> y[3][k] - 1),2) ;
                    else
                        sum[j] += pow(c -> y[3][k],2) ;
                }
            }
            
            k = 0;
            for(j = 1; j < NODES3; j++)
                if(sum[k] > sum[j])
                    k = j;
           
            /* 
            k = 0;
            for(j = 1; j < NODES3; j++)
                if(c -> y[3][j] > c -> y[3][k])
                    k = j;
            */

            if(c -> y[0][784] != k)
                erros += 1;

            counter++;           
        }
    }
    erros = (100.0 * erros)/counter;
    printf("Erro: %.2f%%\n\n", erros);

    free(c);
    free(sum);
    free(entradateste);
    fclose(arquivomap);
    fclose(testefile);

    return erros;
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

