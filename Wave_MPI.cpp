#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;

class wave_class
{
public:
	int row_allocate;
	int id_up;
	int id_down;
	static void buildMPIType();
	static MPI_Datatype MPI_type;
};

MPI_Datatype wave_class::MPI_type;

void wave_class::buildMPIType()
{
	int block_lengths[3];
	MPI_Aint displacements[3];
	MPI_Aint addresses[3], add_start;
	MPI_Datatype typelist[3];

	wave_class temp;

	typelist[0] = MPI_INT;
	block_lengths[0] = 1;
	MPI_Get_address(&temp.row_allocate, &addresses[0]);

	typelist[1] = MPI_INT;
	block_lengths[1] = 1;
	MPI_Get_address(&temp.id_up, &addresses[1]);

	typelist[2] = MPI_INT;
	block_lengths[2] = 1;
	MPI_Get_address(&temp.id_down, &addresses[2]);

	MPI_Get_address(&temp, &add_start);
	for (int i = 0; i < 3; i++) displacements[i] = addresses[i] - add_start;

	MPI_Type_create_struct(3, block_lengths, displacements, typelist, &MPI_type);
	MPI_Type_commit(&MPI_type);
}

vector<vector<double> > grid, new_grid, old_grid;
vector <vector<double> > new_grid_each_process;
vector <vector<double> > grid_each_process;
vector <vector<double> > old_grid_each_process;
int imax = 100, jmax = 100;
double t_max = 30.0;
double t, t_out = 0.0, dt_out = 0.04, dt;
double y_max = 10.0, x_max = 10.0, dx, dy;
double c = 1;

void grid_to_file(int out)
{
	//Write the output for a single time step to file
	stringstream fname;
	fstream f1;
	fname << "./out2/output" << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i < imax; i++)
	{
		for (int j = 0; j < jmax; j++)
			f1 << grid[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
}

//Do a single time step
//on each process, only calculate the rows which are assigned to this process
void do_iteration(int row_allocate)
{
	//Calculate the new displacement for all the points not on the boundary of the domain
	//Note that in parallel the edge of processor's region is not necessarily the edge of the domain
	for (int i = 1; i < row_allocate - 1; i++)
		for (int j = 1; j < jmax - 1; j++)
			new_grid_each_process[i][j] = pow(dt * c, 2.0) * ((grid_each_process[i + 1][j] - 2.0 * grid_each_process[i][j] + \
			grid_each_process[i - 1][j]) / pow(dx, 2.0) + (grid_each_process[i][j + 1] - 2.0 * grid_each_process[i][j] + \
			grid_each_process[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid_each_process[i][j] - old_grid_each_process[i][j];

	//Implement boundary conditions - This is a Neumann boundary that I have implemented
	for (int i = 0; i < row_allocate; i++)
	{
		new_grid_each_process[i][0] = new_grid_each_process[i][1];
		new_grid_each_process[i][jmax - 1] = new_grid_each_process[i][jmax - 2];
	}

	for (int j = 0; j < jmax; j++)
	{
		new_grid_each_process[0][j] = new_grid_each_process[1][j];
		new_grid_each_process[row_allocate-1][j] = new_grid_each_process[row_allocate-2][j];
	}

	t += dt;

	//Note that I am not copying data between the grids, which would be very slow, but rather just swapping pointers
	old_grid_each_process.swap(new_grid_each_process);
	old_grid_each_process.swap(grid_each_process);
}

void communication(double *tmp1, double *tmp2, double *tmp3, double *tmp4,int id_up, int id_down, MPI_Request * request, int row_allocate)
{
        MPI_Isend(tmp1, jmax, MPI_DOUBLE, id_up, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Isend(tmp2, jmax, MPI_DOUBLE, id_down, 1, MPI_COMM_WORLD, &request[1]);
		MPI_Isend(tmp3, jmax, MPI_DOUBLE, id_up, 2, MPI_COMM_WORLD, &request[2]);
        MPI_Isend(tmp4, jmax, MPI_DOUBLE, id_down, 3, MPI_COMM_WORLD, &request[3]);

        MPI_Irecv(&grid_each_process[0][0], jmax, MPI_DOUBLE, id_up, 1, MPI_COMM_WORLD, &request[4]);
        MPI_Irecv(&grid_each_process[row_allocate-1][0], jmax, MPI_DOUBLE, id_down, 0, MPI_COMM_WORLD, &request[5]);
		MPI_Irecv(&old_grid_each_process[0][0], jmax, MPI_DOUBLE, id_up, 3, MPI_COMM_WORLD, &request[6]);
        MPI_Irecv(&old_grid_each_process[row_allocate-1][0], jmax, MPI_DOUBLE, id_down, 2, MPI_COMM_WORLD, &request[7]);

        MPI_Waitall(8, request, MPI_STATUS_IGNORE);
}

int * get_rec_cnt(int processNum, int roweach, int rowlast)
{
	int split_num[processNum];
	for(int i=0;i<processNum-1;i++)
		split_num[i] = roweach;
	split_num[processNum-1] = rowlast;
	int * cnt_recv = new int [processNum];
    for (int i = 0; i<processNum; i++)
        {
            cnt_recv[i] = split_num[i] * jmax;
        }
	return cnt_recv;
}

int * get_displacement(int processNum, int roweach)
{
	int * displacement = new int [processNum];
    for (int i = 0; i<processNum; i++)
        {
            displacement[i] = i * roweach * jmax;
        }
	return displacement;
}

int main(int argc, char *argv[])
{
	old_grid.resize(imax, vector<double>(jmax));
	grid.resize(imax, vector<double>(jmax));
	new_grid.resize(imax, vector<double>(jmax));

	dx = x_max / ((double)imax - 1);
	dy = y_max / ((double)imax - 1);

	t = 0.0;

	dt = 0.1 * min(dx, dy) / c;

	int out_cnt = 0, it = 0;

	//sets half sinusoidal intitial disturbance - this is a bit brute force and it can be done more elegantly
	double r_splash = 1.0;
	double x_splash = 3.0;
	double y_splash = 3.0;
	for (int i = 1; i < imax - 1; i++)
		for (int j = 1; j < jmax - 1; j++)
		{
			double x = dx * i;
			double y = dy * j;

			double dist = sqrt(pow(x - x_splash, 2.0) + pow(y - y_splash, 2.0));

			if (dist < r_splash)
			{
				double h = 5.0*(cos(dist / r_splash * M_PI) + 1.0);

				grid[i][j] = h;
				old_grid[i][j] = h;
			}
		}

	grid_to_file(out_cnt);
	out_cnt++;
	t_out += dt_out;

	int id, processNum;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &processNum);
	wave_class::buildMPIType();
	wave_class data;
//divide the domain into strips, calculate the average row for each process, if it is not divisible
//the last process will have the average add the mod
	int row_each = imax / processNum;
	int row_last = imax - row_each * (processNum - 1);

//allocate the memory,the first and the last process will add only one addition row to communicate
//while all the other processes will add two rows to communicate
	if(id == processNum-1)
	{
		data.row_allocate = row_last+1;
		new_grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		old_grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		for(int i=1; i<data.row_allocate; i++)
		{
			for(int j=0; j<jmax; j++)
			{
			grid_each_process[i][j] = grid[row_each*id+i-1][j];
			old_grid_each_process[i][j] = grid[row_each*id+i-1][j];
			}
		}
	}
	else if(id == 0)
	{
		data.row_allocate = row_each+1;
		new_grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		old_grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		for(int i=0; i<data.row_allocate-1; i++)
		{
			for(int j=0; j<jmax; j++)
			{
			grid_each_process[i][j] = grid[row_each*id+i][j];
			old_grid_each_process[i][j] = grid[row_each*id+i][j];
			}
		}
	}
	else
	{
		data.row_allocate = row_each+2;
		new_grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		old_grid_each_process.resize(data.row_allocate,vector<double>(jmax));
		for(int i=1; i<data.row_allocate-1; i++)
		{
			for(int j=0; j<jmax; j++)
			{
			grid_each_process[i][j] = grid[row_each*id+i-1][j];
			old_grid_each_process[i][j] = grid[row_each*id+i-1][j];
			}
		}
	}

//define neighbour process
    if (id > 0)	data.id_up = id - 1;
    else	data.id_up = MPI_PROC_NULL;
    if ( id < processNum - 1 ) data.id_down = id + 1;
    else	data.id_down = MPI_PROC_NULL;

	cout <<id<<" "<<data.id_up<<" "<<data.id_down<<" "<<data.row_allocate<<endl;
	MPI_Request * request = new MPI_Request[8];

	while (t < t_max)
	{        
		double *tmp1 = new double [jmax];
        double *tmp2 = new double [jmax];
		double *tmp3 = new double [jmax];
        double *tmp4 = new double [jmax];
        for (int i = 0; i < jmax; i++)
        {
            tmp1[i] = grid_each_process[1][i];
            tmp2[i] = grid_each_process[data.row_allocate-2][i];
			tmp3[i] = old_grid_each_process[1][i];
			tmp4[i] = old_grid_each_process[data.row_allocate-2][i];
        }
        
       communication(tmp1, tmp2, tmp3, tmp4, data.id_up, data.id_down, request, data.row_allocate);

		//print out test
		// if(id == 0 && it == 1) 
		// {
		// 	for(int i=0; i<row_allocate; i++)
		// 	{
		// 		for(int j=0; j<jmax; j++)
		// 			cout<<grid_each_process[i][j]<<"\t";
		// 		cout<<endl;
		// 	}
		// 	cout<<endl;
		// }

        do_iteration(data.row_allocate);
        MPI_Barrier(MPI_COMM_WORLD);
		
//define start_row and end_row ,in order to gather back to the grid
		int send_start_row, send_row_number;
		if(id == 0)
		{
			send_start_row =0;
			send_row_number = data.row_allocate-1;
		}
		else if(id == processNum-1)
		{
			send_start_row = 1;
			send_row_number = data.row_allocate-1;
		}
		else
		{
			send_start_row =1;
			send_row_number = data.row_allocate-2;
		}


		double *buffer_send = new double [send_row_number*jmax];
		int buffer_send_cnt = 0;
		for (int i=send_start_row;i<send_start_row+send_row_number;i++)
		{
			for(int j=0;j<jmax;j++)
				{	
					buffer_send[buffer_send_cnt++] = grid_each_process[i][j];
				}
		}

// gather the grid calculated in each process to the grid in process 0.		
		if(id == 0)
		{
			int * cnt_recv;
			cnt_recv = get_rec_cnt(processNum,row_each,row_last);
            int * displacement;
			displacement = get_displacement(processNum,row_each);
			
			double *buffer_recv = new double [imax*jmax];
			MPI_Gatherv(buffer_send, send_row_number*jmax, MPI_DOUBLE, &grid[0][0], cnt_recv, displacement, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			// for(int i=0; i<imax; i++)
			// {
			// 	for(int j=0; j<jmax; j++)
			// 		cout<<grid[i][j]<<"\t";
			// 	cout<<endl;
			// }
			// cout<<endl;

			if (t_out <= t)
			{
				cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << endl;
				grid_to_file(out_cnt);
				out_cnt++;
				t_out += dt_out;
			}
			delete [] cnt_recv;
			delete [] displacement;
		}
		else
		{
			MPI_Gatherv(buffer_send, send_row_number*jmax, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		
		delete [] buffer_send;
		delete [] tmp1;delete [] tmp2;delete [] tmp3;delete [] tmp4;

		it++;
	}
	MPI_Finalize();
	return 0;
}