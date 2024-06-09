#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

enum { N, E, S, W, NE, SE, SW, NW }; // Cardinal points
enum { X, Y }; // Axis
enum { SEND, RECV }; // Send/Recv 

/*Arma un grid de dos dimensiones con la cantidad de celdas cells
  cells debe ser par */
void dimensionalize(int cells, int dims[2])
{
    assert(!(cells%2));
    int n, m;
    n = m = ceil(sqrt(cells));
    while(n*m != cells)
    {
        n--;
        m = cells/n;
    }
    dims[X]=n;
    dims[Y]=m;
}

/*Convierte cordenadas (i, j) globales a (y, x) locales
  si las coordenadas globales no se encuentran en el area local, retorna 0, sino 1 */
int translate(int i, int j, int size[2], int begin[2], int* new_y, int* new_x)
{
    *new_y = (i-begin[Y]); 
    *new_x = (j-begin[X]);
    return (*new_y >= 0 && *new_y < size[Y] && *new_x >= 0 && *new_x < size[X]);
}

int get_relative_rank(MPI_Comm comm, int dx, int dy, const int coords[2], const int dims[2], int* rank)
{
    int to_coords[2] = { coords[X]+dx, coords[Y]+dy };
    if(dims[Y] > to_coords[Y] && to_coords[Y] >= 0)
    	MPI_Cart_rank(comm, to_coords, rank); 
    else
	    *rank = MPI_PROC_NULL;
}

void live(int neighbords, const char* old, char* next)
{
    if (*old == 1) //si tiene 2 o 3 vecinas vivas, sigue viva
        *next = (neighbords == 2 || neighbords == 3) ? 1 : 0;
    else
        //Si está muerta y tiene 3 vecinas vivas revive
        *next = (neighbords == 3) ? 1 : 0;
}

void row_limit_step(char** old, char** nextStep, char* buffer, int rel, int y, int width)
{
    for(int x=1; x<width-1; x++)
    {
     	int cant = old[y][x+1] + old[y][x-1] + old[y-rel][x] + old[y-rel][x-1] + old[y-rel][x+1] + buffer[x] + buffer[x+1] + buffer[x-1];
	    live(cant, &old[y][x], &nextStep[y][x]);
    }
}

void col_limit_step(char** old, char** nextStep, char* buffer, int rel, int x, int height)
{
    for(int y=1; y<height-1; y++)
    {
        int cant = old[y+1][x] + old[y-1][x] + old[y][x-rel] + old[y-1][x-rel] + old[y+1][x-rel] + buffer[y] + buffer[y+1] + buffer[y-1];
	    live(cant, &old[y][x], &nextStep[y][x]);
    }
}

int main(int argc, char *argv[]) 
{
    //Controla que se haya ingresado un argumento en la llamada (el archivo con el  patrón de entrada) 
    if (argc != 2)
    {
        printf("Error: Debe indicar el nombre del archivo de entrada\n");
        return 1;
    }

    //Lectura del encabezado del archivo que contiene el patrón de celdas
    FILE* f = fopen(argv[1], "r");
    if (f == NULL) 
    {
        printf("Error al intentar abrir el archivo.\n");
        return 1;
    }

    int rows, cols, steps;
    if (fscanf(f, "cols %d\nrows %d\nsteps %d\n", &cols, &rows, &steps) != 3)
    {
        printf("Error: formato de archivo incorrecto\n");
        return 1;
    }

    assert(rows==cols);

    MPI_Init(&argc, &argv);

    int ranks;
    MPI_Comm comm_cart;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    const int ndims = 2;     
    //toroidal    
    const int periods[2] = {1, 0};      
    int dims[2];
    dimensionalize(ranks, dims);
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &comm_cart);

    int rank;
    int coords[2];
    MPI_Comm_rank(comm_cart, &rank);
    MPI_Cart_coords(comm_cart, rank, ndims, coords);

    /* *********************************************  
        ARREGLAR ESTA PARTE, 21/5 = 5 + 5 + 5 + 5 + 1
        PERO DEBERIA SER     21/5 = 5 + 4 + 4 + 4 + 4 
     ****************************/
    int size_max[2] = { (int)ceil(cols/dims[X]), (int)ceil((float)rows/dims[Y]) };
    int size[2] = { size_max[X], size_max[Y] };
    
    // last block could be smaller
    if(cols%size[X] && coords[X]==dims[X]-1)
        size[X] = cols%size[X];
    if(rows%size[Y] && coords[Y]==dims[Y]-1)
        size[Y] = rows%size[Y];
    
    // global coord for local(0,0)
    int begin[2] = { coords[X]*size_max[X], coords[Y]*size_max[Y]};

    /**************************************
    /* HASTA ACA IMPACTA,   post condiciones deberian ser size y begin bien calculados 
    /* size_max no sirve para nada, solo lo uso aca
    ***************************************/

    // alloc grid data
    char* grid = (char*)malloc(size[X]*size[Y]*sizeof(char));
    char* grid_next = (char*)malloc(size[X]*size[Y]*sizeof(char));
    char** old = (char**)malloc(size[X]*sizeof(char*));
    char** nextStep = (char**)malloc(size[X]*sizeof(char*));
    for(int i=0; i<size[Y]; i++)
    {
        old[i] = grid+(size[X]*i);   
    	nextStep[i] = grid_next+(size[X]*i);
    }

    char* line = (char*)malloc(size[X]);    
    int i = 0;
    int x, y;
    while (i < rows && fgets(line, size[X], f) != NULL) 
    {
        int line_size = strlen(line) - 1;
        for (int j = 0; j < line_size; j++)
            if(translate(i, j, size, begin, &y, &x))
                old[y][x] = (line[j] == '.') ? 0 : 1;

        /*relleno hacia eje x*/
        translate(i, line_size, size, begin, &y, &x);
        if(y >= 0 && y < size[Y])
        {
            if(x < 0)
                x = 0;
            memset(&old[y][x], 0, size[X]-x);
        }
        i++;
    }

    /*relleno hacia eje y*/
    translate(i, 0, size, begin, &y, &x);
    if(y<0)
        y = 0;
    if(y<size[Y])
    	memset(&old[y][0], 0, size[0]*(size[Y]-y));

    fclose(f); //Se cierra el archivo 
    free(line); //Se libera la memoria utilizada para recorrer el archivo

    // Define row and col buffer types
    MPI_Datatype buffer_type[2];
    for(int i=0; i<2; i++)
    {
    	MPI_Type_vector(1, size[i], 0, MPI_CHAR, &buffer_type[i]);
    	MPI_Type_commit(&buffer_type[i]);
    }

    // Alloc recv/send limit buffers
    char* buffer[4][2]; 
    for(int cardinal=0; cardinal<4; cardinal++)
	for(int side=0; side<2; side++)
	{
	    buffer[cardinal][side] = (char*)malloc(sizeof(char)*size[cardinal%2]);
	    memset(buffer[cardinal][side], 0, size[cardinal%2]);
	}

    char corner[8][2];
    int neigh[8];

    MPI_Cart_shift(comm_cart, 0, 1, &neigh[W], &neigh[E]);
    MPI_Cart_shift(comm_cart, 1, 1, &neigh[N], &neigh[S]);    
    get_relative_rank(comm_cart, -1, -1, coords, dims, &neigh[NW]);
    get_relative_rank(comm_cart, +1, +1, coords, dims, &neigh[SE]);
    get_relative_rank(comm_cart, +1, -1, coords, dims, &neigh[NE]);
    get_relative_rank(comm_cart, -1, +1, coords, dims, &neigh[SW]);

    MPI_Request request[2][8];
    MPI_Status status[2][8];

    for(int c=0; c<steps; c++)
    {
	    //Fill send buffers
        for(int x=0; x<size[X]; x++)
   	    {
     	    buffer[N][SEND][x] = old[0][x];
	        buffer[S][SEND][x] = old[size[Y]-1][x];
    	}
    	for(int y=0; y<size[Y]; y++)
   	    {
	        buffer[W][SEND][y] = old[y][0];
 	        buffer[E][SEND][y] = old[y][size[X]-1];
 	    }

    	corner[NE][SEND] = old[0][size[X]-1];
   	    corner[NW][SEND] = old[0][0];
    	corner[SE][SEND] = old[size[Y]-1][size[X]-1];
    	corner[SW][SEND] = old[size[Y]-1][0];

	    // Send limit rows/cols
    	for(int cardinal=0; cardinal<4; cardinal++)
	        if(neigh[cardinal] != MPI_PROC_NULL)
    	 	    MPI_Isend(buffer[cardinal][SEND], 1, buffer_type[cardinal%2], neigh[cardinal], 0, comm_cart, &request[SEND][cardinal]); 
	        else
		        request[SEND][cardinal] = MPI_REQUEST_NULL;

	    // Send corners
    	for(int cardinal=4; cardinal<8; cardinal++)
	        if(neigh[cardinal] != MPI_PROC_NULL)
                MPI_Isend(&corner[cardinal][SEND], 1, MPI_CHAR, neigh[cardinal], 0, comm_cart, &request[SEND][cardinal]); 
	        else
		        request[SEND][cardinal] = MPI_REQUEST_NULL;

	    // Receive limit rows/cols
    	for(int cardinal=0; cardinal<4; cardinal++)
	        if(neigh[cardinal] != MPI_PROC_NULL)
                MPI_Irecv(buffer[cardinal][RECV], 1, buffer_type[cardinal%2], neigh[cardinal], 0, comm_cart, &request[RECV][cardinal]);
	        else
		        request[RECV][cardinal] = MPI_REQUEST_NULL;

	    // Receive corners
   	    for(int cardinal=4; cardinal<8; cardinal++)
  	        if(neigh[cardinal] != MPI_PROC_NULL)
	    	    MPI_Irecv(&corner[cardinal][RECV], 1, MPI_CHAR, neigh[cardinal], 0, comm_cart, &request[RECV][cardinal]); 
	        else
		        request[RECV][cardinal] = MPI_REQUEST_NULL;

	    // Calc inner grid
        for (int i = 1; i < size[Y]-1; i++)
        {
            for (int j = 1; j < size[X]-1; j++)
            {
                //Suma las celdas vecinas para saber cuantas están vivas
                int cant1 = old[i-1][j-1] + old[i-1][j] + old[i-1][j+1] + old[i][j-1] + old[i][j+1] + old[i+1][j-1] + old[i+1][j] + old[i+1][j+1];
                live(cant1, &old[i][j], &nextStep[i][j]);
            }
        }

	    MPI_Waitall(8, request[RECV], status[RECV]);

	    // Calc limits
   	    row_limit_step(old, nextStep, buffer[N][RECV], -1, 0,         size[X]);
   	    row_limit_step(old, nextStep, buffer[S][RECV], +1, size[Y]-1, size[X]);
   	    col_limit_step(old, nextStep, buffer[W][RECV], -1, 0,         size[Y]);
   	    col_limit_step(old, nextStep, buffer[E][RECV], +1, size[X]-1, size[Y]);

	    // Calc corners
   	    live(old[0][1] + old[1][0] + old[1][1] + buffer[W][RECV][0] + buffer[W][RECV][1] + buffer[N][RECV][0] + buffer[N][RECV][1] + corner[NW][RECV], &old[0][0], &nextStep[0][0]);
   	    live(old[0][size[X]-2] + old[1][size[X]-1] + old[1][size[X]-2] + buffer[E][RECV][0] + buffer[E][RECV][1] + buffer[N][RECV][size[X]-1] + buffer[N][RECV][size[X]-2] + corner[NE][RECV], &old[0][size[X]-1], &nextStep[0][size[X]-1]);
   	    live(old[size[Y]-1][1] + old[size[Y]-1][0] + old[size[Y]-2][1] + buffer[W][RECV][size[Y]-1] + buffer[W][RECV][size[Y]-2] + buffer[S][RECV][0] + buffer[S][RECV][1] + corner[SW][RECV], &old[size[Y]-1][0], &nextStep[size[Y]-1][0]);
   	    live(old[size[Y]-1][size[X]-2] + old[size[Y]-2][size[X]-1] + old[size[Y]-2][size[X]-2] + buffer[E][RECV][size[Y]-1] + buffer[E][RECV][size[Y]-2] + buffer[S][RECV][size[X]-1] + buffer[S][RECV][size[X]-2] + corner[SE][RECV], &old[size[Y]-1][size[X]-1], &nextStep[size[Y]-1][size[X]-1]);

	    char** aux = old;
	    old = nextStep;
	    nextStep = aux;

        MPI_Waitall(8, request[SEND], status[SEND]);
    }

    //store data
    char out_name[20];
    snprintf(out_name, 20, "subgrid_%d_%d.out", coords[Y], coords[X]);
    FILE* out_file = fopen(out_name, "w");
    for (int y = 0; y < size[Y]; y++) 
    {
        for (int x = 0; x < size[X]; x++) 
        {
            fprintf(out_file, "%c", old[y][x] == 1 ? 'O' : '.');
        }
        fprintf(out_file, "\n");
    }

    fclose(out_file);
    MPI_Finalize();
}
