// quick host debug: per-term breakdown of the inter score for a pose
#include <cstdio>
#include <cmath>
#include <vector>
#include "../gpdock/src/smina_types.h"

static float ggaussian_f(float x, float w){ return std::exp(-x*x/(w*w)); }
static float gslope(float xb,float xg,float x){ if(xb<xg){ if(x<=xb)return 0; if(x>=xg)return 1;} else {if(x>=xb)return 0; if(x<=xg)return 1;} return (x-xb)/(xg-xb);}

int main(){
  // read receptor + ligand, sum terms
  // parse minimal PDB (heavy atoms)
  auto parse=[&](const char*fn, std::vector<float>&xyz, std::vector<unsigned char>&t, bool isrec){
    FILE*f=fopen(fn,"r"); if(!f){fprintf(stderr,"no %s\n",fn);exit(1);}
    char line[256];
    // we already have typed? No - re-derive element only, type from file is gone.
    // Instead read element from cols 77-78 (or atom name) - but we need types.
    // For debugging we'll just parse element and map to a coarse type.
    while(fgets(line,256,f)){
      if(!(strncmp(line,"ATOM",4)==0||strncmp(line,"HETATM",6)==0)) continue;
      char el[3]={0};
      if(strlen(line)>=78){ el[0]=line[76]; if(line[77]!=' '&&line[77]!='\n')el[1]=line[77]; }
      if(el[0]==' '||el[0]=='\n')el[0]=0;
      el[0]=toupper(el[0]); if(el[1])el[1]=toupper(el[1]);
      if(el[0]=='H')continue;
      float x=atof(line+30),y=atof(line+38),z=atof(line+46);
      xyz.push_back(x);xyz.push_back(y);xyz.push_back(z);
      unsigned char tt;
      if(el[0]=='C')tt=SMT_C_HPHOB;
      else if(el[0]=='N')tt=SMT_N_D;
      else if(el[0]=='O')tt=SMT_O_A;
      else if(el[0]=='S')tt=SMT_S;
      else if(el[0]=='P')tt=SMT_P;
      else if(el[0]=='F')tt=SMT_F;
      else if(el[0]=='C'&&el[1]=='l')tt=SMT_CL;
      else tt=SMT_C_HPHOB;
      t.push_back(tt);
    }
    fclose(f);
  };
  std::vector<float>rx,ry,rz;std::vector<unsigned char>rt;
  std::vector<float>lx,ly,lz;std::vector<unsigned char>lt;
  // parse returns flattened xyz; split
  { std::vector<float>xyz;std::vector<unsigned char>t; parse("rec_pdb",xyz,t,true); for(size_t i=0;i<xyz.size();i+=3){rx.push_back(xyz[i]);ry.push_back(xyz[i+1]);rz.push_back(xyz[i+2]);rt.push_back(t[i/3]);} }
  { std::vector<float>xyz;std::vector<unsigned char>t; parse("lig_pdb",xyz,t,false); for(size_t i=0;i<xyz.size();i+=3){lx.push_back(xyz[i]);ly.push_back(xyz[i+1]);lz.push_back(xyz[i+2]);lt.push_back(t[i/3]);} }
  printf("rec atoms=%zu lig atoms=%zu\n",rx.size(),lx.size());
  float t1e=0,t2e=0,t3e=0,t4e=0,t5e=0; long npairs=0, ncut=0;
  for(size_t i=0;i<lx.size();i++){
    for(size_t j=0;j<rx.size();j++){
      float dx=lx[i]-rx[j],dy=ly[i]-ry[j],dz=lz[i]-rz[j];
      float d2=dx*dx+dy*dy+dz*dz; if(d2>64)continue; float r=sqrtf(d2);
      float d=r-goptimal_distance(lt[i],rt[j]);
      float e=0;
      float a1=-0.035579f*ggaussian_f(d-0,0.5); t1e+=a1;
      float a2=-0.005156f*ggaussian_f(d-3,2);   t2e+=a2;
      float a3=0.840245f*((d<0)?d*d:0);         t3e+=a3;
      float a4=0; if(gxs_hydrophobic(lt[i])&&gxs_hydrophobic(rt[j])) a4=-0.035069f*gslope(1.5,0.5,d); t4e+=a4;
      float a5=0; if(gxs_h_bond_possible(lt[i],rt[j])) a5=-0.587439f*gslope(0,-0.7,d); t5e+=a5;
      e+=a1+a2+a3+a4+a5; npairs++;
      (void)e;
    }
  }
  printf("pairs within cutoff=%ld\n",npairs);
  printf("gauss0(0,0.5): %+.3f\n",t1e);
  printf("gauss2(3,2)  : %+.3f\n",t2e);
  printf("repulsion    : %+.3f\n",t3e);
  printf("hydrophobic  : %+.3f\n",t4e);
  printf("hbond        : %+.3f\n",t5e);
  printf("TOTAL        : %+.3f\n",t1e+t2e+t3e+t4e+t5e);
  return 0;
}
