/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "compute_tgt_atom.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "bond.h" 
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "math_const.h"
#include "error.h"
#include "random_mars.h"

using namespace LAMMPS_NS;
double *LAMMPS_NS::ComputeTgtAtom::randphase=NULL;
double *LAMMPS_NS::ComputeTgtAtom::poisson_delt=NULL;
double *LAMMPS_NS::ComputeTgtAtom::next_time=NULL;

/* ---------------------------------------------------------------------- */

ComputeTgtAtom::ComputeTgtAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal compute tgt/local command");

  if (atom->avec->bonds_allow == 0)
    error->all(FLERR,"Compute tgt/local used when bonds are not allowed");

  fstr = force->numeric(FLERR,arg[3]);
  omega = force->numeric(FLERR,arg[4]);
  tau = force->numeric(FLERR,arg[5]);
  seed = force->numeric(FLERR,arg[6]);
  randtgt=NULL; 
  int i; 


  local_flag = 1;
  size_peratom_cols=3; 
  create_attribute = 1; 
  nvalues = narg - 3;
  if (nvalues == 1) size_local_cols = 0;
  else size_local_cols = nvalues;

  bstyle = new int[nvalues];

  nvalues = 0;
  
  // set singleflag if need to call bond->single()

  singleflag = 0;
 
  peratom_flag = 1;	
  nmax = 0;
  factive = NULL; 
}

/* ---------------------------------------------------------------------- */

ComputeTgtAtom::~ComputeTgtAtom()
{
  delete randtgt;
  delete [] bstyle;
  //delete [] randphase; 
  memory->destroy(factive); 
}

/* ---------------------------------------------------------------------- */

void ComputeTgtAtom::init()
{
  if (force->bond == NULL)
    error->all(FLERR,"No bond style is defined for compute bond/local");

  // do initial memory allocation so that memory_usage() is correct
/*
  ncount = compute_bonds(0);
  if (ncount > nmax) reallocate(ncount);
  size_local_rows = ncount;*/
}

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/*
void ComputeTgtAtom::compute_local()
{
  invoked_local = update->ntimestep;

  // count local entries and compute bond info

  ncount = compute_bonds(0);
  if (ncount > nmax) reallocate(ncount);
  size_local_rows = ncount;
  ncount = compute_bonds(1);
}*/

/* ----------------------------------------------------------------------*/
//
//void ComputeTgtAtom::init_list(int id, NeighList *ptr)
//{
//  list = ptr;
//}

/* ---------------------------------------------------------------------- */
void ComputeTgtAtom::compute_peratom()
{
	int i,tau_steps,delt_steps,next_steps; 
	double MY_2PI=LAMMPS_NS::MathConst::MY_2PI;
	double tvalue,next_rand;
	tau_steps=(int)(tau/(update->dt));


	tvalue = update->atime + (update->ntimestep-update->atimestep)*update->dt;

	tagint maxmol = 0;
	if (atom->molecule_flag) {
		for (i = 0; i < atom->nlocal; i++) maxmol = MAX(atom->molecule[i],maxmol);
		tagint maxmol_all;
		MPI_Allreduce(&maxmol,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
		maxmol = maxmol_all;
	}

	if (randtgt==NULL){
		randtgt = new RanMars(lmp,seed);}

	if(!randphase){ 
		randphase = new double[maxmol];
	}
	if(!poisson_delt){ 
		poisson_delt = new double[maxmol];
		for(i=0;i<maxmol;i++){poisson_delt[i]=0.0;}
	}

	if(!next_time){ 
		next_time = new double[maxmol];
		for(i=0;i<maxmol;i++){next_time[i]=0.01;}
	}

	if (comm->me==0){ 


		//	fprintf(stderr," Entered interval delt_steps = %d, next_steps= %d\n",delt_steps,next_steps);	
		for(i=0;i<maxmol;i++){

			poisson_delt[i] = poisson_delt[i] + (double)update->dt; 
			delt_steps=(int)(poisson_delt[i]/(update->dt));
			//fprintf(stderr,"poissn_delt = %lf, next_steps= %d, delt_steps = %d rand_tgt %lf \n",poisson_delt,next_steps,delt_steps,randtgt->uniform());	
			next_steps=(int)(next_time[i]/(update->dt))+1;
			//fprintf(stderr,"i=%d, next_time= %lf\n",i,next_time[i]);
			if  (delt_steps==next_steps){

				randphase[i] =MY_2PI*(randtgt->uniform());
				poisson_delt[i]=0.0;
				next_rand = randtgt->uniform(); 
				if (next_rand<0.00000000001) {next_time[i]=0.0;} 
				else {next_time[i] = -logl(next_rand)*tau;	}

			}  
		}
	}	
	MPI_Bcast(randphase,maxmol,MPI_DOUBLE,0,world);

	if (atom->nmax > nmax) {
		memory->destroy(factive);
		nmax = atom->nmax;
		memory->create(factive,nmax,3,"factive/atom:factive");
		array_atom = factive;
	}

	int m,n,nb,atom1,atom2,imol,iatom,iatom2,btype;
	tagint tagprev;
	double delx,dely,delz,rsq,delx1,dely1,delz1;
	double *ptr;
	//  double *randphase;  // defined in .h file 

	double **x = atom->x;
	tagint *tag = atom->tag;
	tagint *molecule = atom->molecule; 
	int *num_bond = atom->num_bond;
	tagint **bond_atom = atom->bond_atom;
	int **bond_type = atom->bond_type;
	int *mask = atom->mask;

	int *molindex = atom->molindex;
	int *molatom = atom->molatom;
	Molecule **onemols = atom->avec->onemols;

	int nlocal = atom->nlocal;
	int newton_bond = force->newton_bond;
	int molecular = atom->molecular;

	Bond *bond = force->bond;
	double eng,fbond,phase;


	// maxmol = largest molecule tag across all existing atoms


	m = n = 0;
	for (atom1 = 0; atom1 < nlocal; atom1++) {
		if (!(mask[atom1] & groupbit)) continue;

		if (molecular == 1) nb = num_bond[atom1];
		else {
			if (molindex[atom1] < 0) continue;
			imol = molindex[atom1];
			iatom = molatom[atom1];
			nb = onemols[imol]->num_bond[iatom];
		}
		delx = dely = delz = 0.0;
		for (i = 0; i < nb; i++) {
			if (molecular == 1) {
				btype = bond_type[atom1][i];
				atom2 = atom->map(bond_atom[atom1][i]);
			} else {
				tagprev = tag[atom1] - iatom - 1;
				btype = onemols[imol]->bond_type[iatom][i];
				atom2 = atom->map(onemols[imol]->bond_atom[iatom][i]+tagprev);
				iatom2 = molatom[atom2];
			}

			if (atom2 < 0 || !(mask[atom2] & groupbit)) continue;
			//if (newton_bond == 0 && tag[atom1] > tag[atom2]) continue;
			if (btype == 0) continue;

			if (iatom  > iatom2){	
				delx1 = x[atom1][0] - x[atom2][0];
				dely1 = x[atom1][1] - x[atom2][1];
				delz1 = x[atom1][2] - x[atom2][2];
			} else {
				delx1 = x[atom2][0] - x[atom1][0];
				dely1 = x[atom2][1] - x[atom1][1];
				delz1 = x[atom2][2] - x[atom1][2];
			}
			domain->minimum_image(delx1,dely1,delz1);
			delx+=delx1;
			dely+=dely1;
			delz+=delz1;

		}
		rsq = sqrt(delx*delx + dely*dely + delz*delz);

		if (rsq > 0){
			delx /= rsq;
			dely /= rsq;
			delz /= rsq;
		}

		/*factive[atom1][0]=fstr*sin(omega*tvalue)*delx;  
		  factive[atom1][1]=fstr*sin(omega*tvalue)*dely;  
		  factive[atom1][2]=fstr*sin(omega*tvalue)*delz; */

		phase= randphase[molecule[atom1]-1];//(double)molecule[atom1]/(double)maxmol*MY_2PI; 

		factive[atom1][0]=fstr*(2.0*(1.0-(double)signbit(sin(phase)))-1.0)*delx;  
		factive[atom1][1]=fstr*(2.0*(1.0-(double)signbit(sin(phase)))-1.0)*dely;  
		factive[atom1][2]=fstr*(2.0*(1.0-(double)signbit(sin(phase)))-1.0)*delz;  

	}

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
   ------------------------------------------------------------------------- */

double ComputeTgtAtom::memory_usage()
{
	double bytes = nmax*3 * sizeof(double);
	return bytes;
}
