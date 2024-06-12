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
static inline __attribute__((always_inline)) void dimensionalize (int cells, int dims[2])
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

static inline __attribute__((always_inline)) void distribute(const int world[2], const int dims[2], const int coords[2], int size[2], int begin[2])
{
    int base_size[2] = { world[X] / dims[X], world[Y] / dims[Y] };
    int extra_cols = world[X] % dims[X];
    int extra_rows = world[Y] % dims[Y];

    size[X] = base_size[X];
    size[Y] = base_size[Y];

    if (coords[X] < extra_cols) 
        size[X]++;
    if (coords[Y] < extra_rows) 
        size[Y]++;

    begin[X] = coords[X] * base_size[X] + (coords[X] < extra_cols ? coords[X] : extra_cols);
    begin[Y] = coords[Y] * base_size[Y] + (coords[Y] < extra_rows ? coords[Y] : extra_rows);
};


/*Convierte cordenadas (i, j) globales a (y, x) locales
  si las coordenadas globales no se encuentran en el area local, retorna 0, sino 1 */
static inline __attribute__((always_inline)) int translate(int i, int j, int size[2], int begin[2], int* new_y, int* new_x)
{
    *new_y = (i-begin[Y]); 
    *new_x = (j-begin[X]);
    return (*new_y >= 0 && *new_y < size[Y] && *new_x >= 0 && *new_x < size[X]);
}

static inline __attribute__((always_inline)) void get_rank(MPI_Comm comm, int x, int y, int* rank)
{
    int to_coords[2] = { x, y };
    MPI_Cart_rank(comm, to_coords, rank); 
}

static inline __attribute__((always_inline)) void live(int neighbords, const char* old, char* next)
{
    if (*old == 1) //si tiene 2 o 3 vecinas vivas, sigue viva
        *next = (neighbords == 2 || neighbords == 3) ? 1 : 0;
    else
        //Si está muerta y tiene 3 vecinas vivas revive
        *next = (neighbords == 3) ? 1 : 0;
}

static inline __attribute__((always_inline)) void row_limit_step(char** old, char** nextStep, char* buffer, int rel, int y, int width)
{
    for(int x=1; x<width-1; x++)
    {
     	int cant = old[y][x+1] + old[y][x-1] + old[y-rel][x] + old[y-rel][x-1] + old[y-rel][x+1] + buffer[x] + buffer[x+1] + buffer[x-1];
	    live(cant, &old[y][x], &nextStep[y][x]);
    }
}

static inline __attribute__((always_inline)) void col_limit_step(char** old, char** nextStep, char* buffer, int rel, int x, int height)
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

    int steps;
    int world[2];

    if (fscanf(f, "cols %d\nrows %d\nsteps %d\n", &world[X], &world[Y], &steps) != 3)
    {
        printf("Error: formato de archivo incorrecto\n");
        return 1;
    }
    assert(world[X]==world[Y]);

    MPI_Init(&argc, &argv);

    int ranks;
    MPI_Comm comm_cart;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    const int ndims = 2;     
    //toroidal    
    const int periods[2] = {1, 1};
    int dims[2] = {0, 0};
   //MPI_Dims_create(ranks, ndims, dims); // Reemplaza a dimentionalize(ranks,dims)
    dimensionalize(ranks, dims);
 
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &comm_cart);

    int rank;
    int coords[2];
    MPI_Comm_rank(comm_cart, &rank);
    MPI_Cart_coords(comm_cart, rank, ndims, coords);

    int size[2];
    int begin[2];
    distribute(world, dims, coords, size, begin);

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
    while (i < world[Y] && fgets(line, size[X], f) != NULL) 
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

    // Define col buffer types
    MPI_Datatype col_type;
    MPI_Type_vector(size[Y], 1, size[X], MPI_CHAR, &col_type);
    MPI_Type_commit(&col_type);

    // Alloc recv limit buffers
    char* buffer[4]; 
    for(int cardinal=0; cardinal<4; cardinal++)
        buffer[cardinal] = (char*)malloc(sizeof(char)*size[cardinal%2]);
        
    char corner[8];
    int neigh[8];

    MPI_Cart_shift(comm_cart, 0, 1, &neigh[W], &neigh[E]);
    MPI_Cart_shift(comm_cart, 1, 1, &neigh[N], &neigh[S]);    
    get_rank(comm_cart, coords[X]-1, coords[Y]-1, &neigh[NW]);
    get_rank(comm_cart, coords[X]+1, coords[Y]+1, &neigh[SE]);
    get_rank(comm_cart, coords[X]+1, coords[Y]-1, &neigh[NE]);
    get_rank(comm_cart, coords[X]-1, coords[Y]+1, &neigh[SW]);

    MPI_Request request[2][8];
    MPI_Status status[2][8];

    request[SEND][W] = MPI_REQUEST_NULL;
    request[SEND][E] = MPI_REQUEST_NULL;
    request[RECV][W] = MPI_REQUEST_NULL;
    request[RECV][E] = MPI_REQUEST_NULL;

    for(int c=0; c<steps; c++)
    {
        MPI_Isend(&old[0][0], size[X], MPI_CHAR, neigh[N], 0, comm_cart, &request[SEND][N]);
        MPI_Isend(&old[size[Y]-1][0], size[X], MPI_CHAR, neigh[S], 0, comm_cart, &request[SEND][S]);
        MPI_Isend(&old[0][0],         1,       col_type, neigh[W], 0, comm_cart, &request[SEND][W]);
        MPI_Isend(&old[0][size[X]-1], 1,       col_type, neigh[E], 0, comm_cart, &request[SEND][E]);
    
        MPI_Isend(&old[0][0],                 1, MPI_CHAR, neigh[NW], 0, comm_cart, &request[SEND][NW]);
        MPI_Isend(&old[0][size[X]-1],         1, MPI_CHAR, neigh[NE], 0, comm_cart, &request[SEND][NE]);
        MPI_Isend(&old[size[Y]-1][0],         1, MPI_CHAR, neigh[SW], 0, comm_cart, &request[SEND][SW]);
        MPI_Isend(&old[size[Y]-1][size[X]-1], 1, MPI_CHAR, neigh[SE], 0, comm_cart, &request[SEND][SE]);

        MPI_Irecv(buffer[N], size[X], MPI_CHAR, neigh[N], 0, comm_cart, &request[RECV][N]);
        MPI_Irecv(buffer[S], size[X], MPI_CHAR, neigh[S], 0, comm_cart, &request[RECV][S]);
        MPI_Irecv(buffer[W], size[Y], MPI_CHAR, neigh[W], 0, comm_cart, &request[RECV][W]);
        MPI_Irecv(buffer[E], size[Y], MPI_CHAR, neigh[E], 0, comm_cart, &request[RECV][E]);

        MPI_Irecv(&neigh[NW], 1, MPI_CHAR, neigh[NW], 0, comm_cart, &request[RECV][NW]);
        MPI_Irecv(&neigh[NE], 1, MPI_CHAR, neigh[NE], 0, comm_cart, &request[RECV][NE]);
        MPI_Irecv(&neigh[SW], 1, MPI_CHAR, neigh[SW], 0, comm_cart, &request[RECV][SW]);
        MPI_Irecv(&neigh[SE], 1, MPI_CHAR, neigh[SE], 0, comm_cart, &request[RECV][SE]);

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
   	    row_limit_step(old, nextStep, buffer[N], -1, 0,         size[X]);
   	    row_limit_step(old, nextStep, buffer[S], +1, size[Y]-1, size[X]);
   	    col_limit_step(old, nextStep, buffer[W], -1, 0,         size[Y]);
   	    col_limit_step(old, nextStep, buffer[E], +1, size[X]-1, size[Y]);

	    // Calc corners
        live(old[0][1] + old[1][0] + old[1][1] + buffer[W][0] + buffer[W][1] + buffer[N][0] + buffer[N][1] + corner[NW], &old[0][0], &nextStep[0][0]);
   	    live(old[0][size[X]-2] + old[1][size[X]-1] + old[1][size[X]-2] + buffer[E][0] + buffer[E][1] + buffer[N][size[X]-1] + buffer[N][size[X]-2] + corner[NE], &old[0][size[X]-1], &nextStep[0][size[X]-1]);
   	    live(old[size[Y]-1][1] + old[size[Y]-1][0] + old[size[Y]-2][1] + buffer[W][size[Y]-1] + buffer[W][size[Y]-2] + buffer[S][0] + buffer[S][1] + corner[SW], &old[size[Y]-1][0], &nextStep[size[Y]-1][0]);
   	    live(old[size[Y]-1][size[X]-2] + old[size[Y]-2][size[X]-1] + old[size[Y]-2][size[X]-2] + buffer[E][size[Y]-1] + buffer[E][size[Y]-2] + buffer[S][size[X]-1] + buffer[S][size[X]-2] + corner[SE], &old[size[Y]-1][size[X]-1], &nextStep[size[Y]-1][size[X]-1]);

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
